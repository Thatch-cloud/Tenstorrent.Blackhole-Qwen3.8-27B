"""P1a of the upload plan (docs/tp4-fabric-upload.md, section 5): a weight-cache HIT no longer pays the host bf16 transpose.

THE COST. tp_common.shard_w is, in the image (tp_common.py sha256 bb43f0cd..., the bytes the simulator gates pin):

    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(w, dtype=dtype, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim), layout=ttnn.TILE_LAYOUT,
                          memory_config=memory_config, cache_file_name=cache_path)

so every one of the ~384 projection loads transposes its weight on the host - about 49 GB of bf16 copies for the 64 layers, single threaded - BEFORE as_tensor
looks at its cache, and on a cache hit the transposed copy is thrown away: as_tensor (ttnn/ttnn/operations/core.py:628-729 in the pinned tt-metal
9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9) opens `{cache_file_name}_dtype_{dtype}_layout_{layout}.tensorbin` first and returns what it loads
(core.py:716-724); the tensor it was given is touched only when the file is missing or fails to load (core.py:719-727 -> from_torch_and_dump), and its
`preprocess` callback runs only there (core.py:676-677, inside torch_to_ttnn, which only from_torch_and_dump calls on a cached path). docs/loading-diagnostic-results.md measured the same thing on the experiment path (34935654817:
samples land in shard_w's transpose; 34936162975: with the transpose moved into `preprocess`, 320 shard loads and 64 packed MLP loads hit the cache and
the target load took 12.2 s against hundreds).

THE LEVER. QWEN_FAST_LAZY_SHARD_W=1 replaces the module attribute tp_common.shard_w (every call site in the grafted model files reads `tpc.shard_w` at call
time; the FILE tp_common.py is not touched, so the simulator pins hold) with a twin that hands the SAME as_tensor call the SAME arguments and moves the transpose
into `preprocess`. A hit does no host copy; a miss runs `.to(bf16).T.contiguous()` and everything after it exactly as before.

THE CACHE. Nothing about the cache changes, which is the point. The cache file holds the finished device-format tensor (sharded, tilized, packed), i.e. what
as_tensor made AFTER the transpose, and its name is built from the caller's cache path, the dtype and the layout only (core.py:716). The twin passes the
same cache path, dtype and layout, so it opens the same files as the stock loader, and a miss writes the same bytes to the same names. An old cache is therefore
valid for the twin (it is the cache the stock loader made) and a new one is valid for the stock loader: there is no new layout to misread and no version to
bump. (A cache that stored the TRANSPOSED HOST tensor instead would need a new name per layout version; this one never held a host tensor.) The cache key
has never covered the CONTENT of the checkpoint or of the conversion code - a stale file is loaded by the stock path as readily as by this one;
QWEN_FAST_LAZY_SHARD_W_AUDIT=1 checks it for the first tensors (below).

AUDIT. QWEN_FAST_LAZY_SHARD_W_AUDIT=1 (with the lever): for the first QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS loads (default 2) the twin ALSO converts the weight
on the host the stock way (transpose, ShardTensorToMesh, bf8/bf4 pack, no device) and compares its packed bytes shard by shard with what as_tensor returned
(read back with Tensor.host_buffer get_shard, per chip). exact=True means the cache file (or the fresh conversion on a miss) is byte for byte the stock
loader's tensor for this checkpoint. A difference logs a mismatch line and latches the lever off for the rest of the process (later loads call the stock
shard_w); the loaded tensor is the one the stock loader would also have loaded, so the engine start goes on unchanged. It costs two host conversions.

OFF: QWEN_FAST_LAZY_SHARD_W unset or 0 -> arm() installs nothing and tp_common.shard_w is the image's function (test_qwen_lazy_shard holds it).

NOT IN SCOPE (measured, left): the fused qkv / qkvz / gate-up weights are still assembled on the host before shard_w sees them (tpc.prepare_attn_qkv*,
tpc.prepare_gdn_qkv and the torch.cat in gdn/tp.py): about 11 GB of copies a load that this lever does not remove, because the graft builds the fused tensor
eagerly; making those lazy is an edit of the grafted attention/tp.py and gdn/tp.py (graft.sha256 changes), a separate step.
"""

import functools
import hashlib
import inspect
import os
import textwrap

from qwen_device_zeros import RawBytesUnavailable, differing_offsets, shard_bytes

FLAG = 'QWEN_FAST_LAZY_SHARD_W'
AUDIT_FLAG = 'QWEN_FAST_LAZY_SHARD_W_AUDIT'
AUDIT_LOADS_FLAG = 'QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS'
NAMES = (FLAG, AUDIT_FLAG, AUDIT_LOADS_FLAG)
DEFAULT_AUDIT_LOADS = 2
SUMMARY_EVERY = 64

TP_COMMON = 'models.demos.blackhole.qwen36.tt.tp_common'
SHARD_W_PARAMETERS = ('torch_tensor', 'mesh', 'dim', 'memory_config', 'cache_path', 'dtype')
# sha256 of textwrap.dedent(inspect.getsource(tp_common.shard_w)) in the image's tp_common.py (sha256 bb43f0cd...0826a); a tp_common whose function differs is left alone.
SHARD_W_SOURCE_SHA256 = '976189f6ed0822ae8a91c278372355a2a4993e52e364c5790a6959b58344fa65'

ENGAGED = '[PINDIAG] tp4 lazy shard engaged'
LOADS = '[PINDIAG] tp4 lazy shard loads'
REFUSED = '[PINDIAG] tp4 lazy shard refused'
AUDIT_LINE = '[PINDIAG] tp4 lazy shard audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 lazy shard audit mismatch'


def _log(template, *values):
    try:
        from loguru import logger
    except ImportError:
        print(template.format(*values), flush=True)
        return
    logger.info(template, *values)


def _switch(environ, name):
    value = environ.get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def enabled(environ=None):
    return _switch(os.environ if environ is None else environ, FLAG)


def flag_problems(env):
    """[problem] for a profile env that names these flags wrongly (not 0/1, an audit without the lever, a count out of range)."""
    problems = []
    for name in (FLAG, AUDIT_FLAG):
        try:
            _switch(env, name)
        except ValueError as error:
            problems.append(str(error))
    value = env.get(AUDIT_LOADS_FLAG)
    if value is not None and not (value.isdigit() and str(int(value)) == value and 1 <= int(value) <= 64):
        problems.append('%s must be an integer in 1..64, got %r' % (AUDIT_LOADS_FLAG, value))
    if not problems:
        if env.get(AUDIT_FLAG) == '1' and env.get(FLAG) != '1':
            problems.append('%s=1 without %s=1: there is no lazy loader to audit' % (AUDIT_FLAG, FLAG))
        if AUDIT_LOADS_FLAG in env and env.get(AUDIT_FLAG) != '1':
            problems.append('%s is set without %s=1: it would do nothing' % (AUDIT_LOADS_FLAG, AUDIT_FLAG))
    return problems


def source_digest(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode('utf-8')).hexdigest()


class Record(object):
    def __init__(self, audit, audit_loads):
        self.calls = self.materialised = self.audited = 0
        self.audit, self.audit_loads = audit, audit_loads
        self.latched = None


def lazy_shard_w(module, original, record, log=_log):
    """The twin of tp_common.shard_w: the same as_tensor call, the transpose in `preprocess`."""
    ttnn, torch = module.ttnn, module.torch

    @functools.wraps(original)
    def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
        """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row. [lazy transpose twin: see qwen_lazy_shard]"""
        if record.latched is not None:
            return original(torch_tensor, mesh, dim, memory_config, cache_path, dtype)
        record.calls += 1
        if record.calls == 1:
            log('{} audit={} audit_loads={}', ENGAGED, 'on' if record.audit else 'off', record.audit_loads if record.audit else 0)

        def preprocess(tensor):
            record.materialised += 1
            return tensor.to(torch.bfloat16).T.contiguous()

        loaded = ttnn.as_tensor(
            torch_tensor,
            dtype=dtype,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            cache_file_name=cache_path,
            preprocess=preprocess,
        )
        if record.audit and record.audited < record.audit_loads:
            record.audited += 1
            audit_load(module, record, loaded, torch_tensor, mesh, dim, dtype, cache_path, log)
        if record.calls % SUMMARY_EVERY == 0:
            log('{} calls={} misses={} hits={} latched={}', LOADS, record.calls, record.materialised, record.calls - record.materialised,
                record.latched or 'no')
        return loaded

    shard_w._qwen_lazy_shard = True
    return shard_w


def audit_load(module, record, loaded, torch_tensor, mesh, dim, dtype, cache_path, log):
    """Compare `loaded` (the cache hit, or the fresh conversion of a miss) with the stock host conversion of the same weight, packed bytes per chip."""
    ttnn, torch = module.ttnn, module.torch
    name = os.path.basename(str(cache_path)) if cache_path else 'uncached'
    try:
        stock = ttnn.from_torch(torch_tensor.to(torch.bfloat16).T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim))
        want, how = shard_bytes(ttnn, mesh, stock)
        got, _ = shard_bytes(ttnn, mesh, ttnn.from_device(loaded))
        if len(got) != len(want):
            raise RawBytesUnavailable('%d shards read back, %d expected' % (len(got), len(want)))
        differing = sum(differing_offsets(left, right, limit=1)[1] for left, right in zip(got, want))
    except Exception as error:  # noqa: BLE001 - an audit that cannot run is not a pass
        record.latched = 'audit of %s raised %s: %s' % (name, type(error).__name__, str(error).strip()[:160])
        log('{} name={} exact=False reason={}', AUDIT_MISMATCH, name, record.latched)
        return False
    if differing:
        record.latched = 'audit of %s: %d bytes differ from the stock conversion' % (name, differing)
        log('{} name={} exact=False differing_bytes={} chips={}', AUDIT_MISMATCH, name, differing, len(want))
        return False
    log('{} exact=True name={} chips={} bytes_per_chip={} raw={}', AUDIT_LINE, name, len(want), len(want[0]) if want else 0, how)
    return True


def install(module, environ=None, log=_log):
    """Replace tp_common.shard_w once its source is the pinned one. -> whether the twin is installed."""
    environ = os.environ if environ is None else environ
    if not _switch(environ, FLAG):
        return False
    original = getattr(module, 'shard_w', None)
    if original is None:
        log('{} reason={}', REFUSED, 'tp_common has no shard_w')
        return False
    if getattr(original, '_qwen_lazy_shard', False):
        return True
    if tuple(inspect.signature(original).parameters) != SHARD_W_PARAMETERS:
        log('{} reason={}', REFUSED, 'shard_w has another signature: the stock loader stays')
        return False
    try:
        digest = source_digest(original)
    except (OSError, TypeError) as error:
        log('{} reason={}', REFUSED, 'shard_w has no readable source (%s): the stock loader stays' % type(error).__name__)
        return False
    if digest != SHARD_W_SOURCE_SHA256:
        log('{} reason={}', REFUSED, 'shard_w is not the pinned function (sha256 %s): the stock loader stays' % digest[:16])
        return False
    audit = _switch(environ, AUDIT_FLAG)
    loads = environ.get(AUDIT_LOADS_FLAG)
    record = Record(audit, int(loads) if loads else DEFAULT_AUDIT_LOADS)
    module.shard_w = lazy_shard_w(module, original, record, log=log)
    module._qwen_lazy_shard_record = record
    return True


def arm(environ=None, on_import=None, log=_log):
    """In every process of a profile that sets QWEN_FAST_LAZY_SHARD_W=1: install the twin when tp_common imports. -> whether it was armed."""
    environ = os.environ if environ is None else environ
    if not _switch(environ, FLAG):
        return False
    on_import(TP_COMMON, lambda module: install(module, environ, log=log))
    return True
