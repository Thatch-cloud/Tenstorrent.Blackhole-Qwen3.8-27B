"""The norms' all-gathers with a named set of per-call options (WP5, F-C1): the twin of the gather inside DistributedNorm.forward.

WHAT THE STACK CALLS. In decode on the (1, 4) mesh DistributedNorm.forward gathers the fractured residual BEFORE the norm (is_distributed_norm is
false for a one-row mesh in decode): ttnn.experimental.all_gather_async(x, persistent_output_buffer=None, dim=3, multi_device_global_semaphore=<the
model's cycled pair>, num_links=tt_ccl.get_num_links(1), topology=args.ccl_topology(), memory_config=<the norm's sharded input config>,
barrier_semaphore=<the model's cycled handle>, chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2, subdevice_id=None). Mode is a
plain Enum, so the config-keyed values (`mode == "decode"` against an Enum member) are never taken and the literals above are the call. 129 of
them a packed pass (2 per layer and the final norm's); the profile (M676) times them at 18.4 us median, 17.4 us on the fastest chip.

WHAT THIS DOES. install() wraps DistributedNorm.forward (a class attribute, as tile_collective_tp wraps ModelBatch.run; the pinned file is not
edited). Outside a block scope the wrapper is the original function. Inside one (the packed verify's 64-row forward, ccl_options_tp.current()) and
in decode it rebinds the module-global `ttnn` of distributed_norm.py to a view whose experimental.all_gather_async is the shim below for the
duration of that one call, and restores it in a finally: no other module, no drafter collective, no other op sees the shim. The shim checks the call
against the census above (refusing, with the reason logged once, every other call: they run unchanged), applies the set's gather options
(ccl_options_tp.ag_overrides) and, for the audit's first N calls of the forward, also runs the call with the model's own values and holds both
results (tile_collective_tp.hold_pair) for the comparison after the replay. The options are routes for the same tiles (the all-gather copies): the audit is
the byte proof on the serving image.

Fail closed: install() refuses to wrap a DistributedNorm.forward whose source is not the pinned one (FORWARD_SHA256), because the shim is only proven
against that source. It wraps nothing when QWEN_FAST_CCL_OPTIONS is unset.

Stdlib only; ttnn is reached through the module the wrapped function sees.
"""

import functools
import hashlib
import importlib
import inspect
import sys

import ccl_options_tp
import tile_collective_tp

NORM_MODULE = 'models.tt_transformers.tt.distributed_norm'
CLASS_NAME = 'DistributedNorm'
# sha256 of the source text of DistributedNorm.forward at tt-metal 9f9cd4fd (the def line through the last line, as inspect.getsource returns it).
FORWARD_SHA256 = 'ea07fcaf23000715da9f0b74d3319c7be8590198006c1fec3850df9f9a5cf8fc'
CENSUS_KEYWORDS = frozenset(['persistent_output_buffer', 'dim', 'multi_device_global_semaphore', 'num_links', 'topology', 'memory_config',
                             'barrier_semaphore', 'chunks_per_sync', 'num_workers_per_link', 'num_buffers_per_channel', 'subdevice_id'])
CENSUS_SERVED = {'chunks_per_sync': 10, 'num_workers_per_link': 2, 'num_buffers_per_channel': 2}
CENSUS_WIDTH = 1280      # 5120 / 4 chips: the fractured residual
CENSUS_LINKS = 2
CENSUS_CHIPS = 4
CENSUS_MESH_SHAPE = (1, 4)


def forward_digest(function):
    """sha256 of a function's source text (what FORWARD_SHA256 pins)."""
    return hashlib.sha256(inspect.getsource(function).encode('utf-8')).hexdigest()


class _View(object):
    """A module as a norm sees it with one attribute replaced: everything else is the module's own."""

    def __init__(self, real, **replaced):
        object.__setattr__(self, '_real', real)
        for name, value in replaced.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_real'), name)


class GatherShim(object):
    """ttnn.experimental.all_gather_async as DistributedNorm.forward sees it inside the scope."""

    def __init__(self, plan, norm, real):
        self.plan = plan
        self.norm = norm
        self.real = real                       # the real ttnn module
        self.gather = real.experimental.all_gather_async

    def refusal(self, tensor, args, kwargs):
        """Why this gather is not the census's (a string naming it), or None when it is."""
        real = self.real
        if args:
            return '%d positional arguments after the tensor' % len(args)
        extra = sorted(set(kwargs) - CENSUS_KEYWORDS)
        missing = sorted(CENSUS_KEYWORDS - set(kwargs))
        if extra or missing:
            return 'keywords differ from the census (extra %s, missing %s)' % (','.join(extra) or '-', ','.join(missing) or '-')
        shape = tuple(tensor.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
            return 'shape %r: leading dimensions are not (1, 1)' % (shape,)
        if shape[2] != self.plan.rows:
            return 'rows %d are not the block\'s %d' % (shape[2], self.plan.rows)
        if shape[3] != CENSUS_WIDTH:
            return 'width %d is not the fractured residual\'s %d' % (shape[3], CENSUS_WIDTH)
        if tensor.dtype != real.bfloat16:
            return 'dtype %r is not bfloat16' % (tensor.dtype,)
        if tensor.layout != real.TILE_LAYOUT:
            return 'layout %r is not tile' % (tensor.layout,)
        if kwargs['dim'] != 3:
            return 'dim %r is not 3' % (kwargs['dim'],)
        if kwargs['persistent_output_buffer'] is not None:
            return 'a persistent output buffer'
        if kwargs['subdevice_id'] is not None:
            return 'a sub-device'
        if kwargs['topology'] != real.Topology.Ring:
            return 'topology %r is not Ring' % (kwargs['topology'],)
        if kwargs['num_links'] != CENSUS_LINKS:
            return 'num_links %r is not %d' % (kwargs['num_links'], CENSUS_LINKS)
        for name, value in sorted(CENSUS_SERVED.items()):
            if kwargs[name] != value:
                return '%s %r is not the model\'s %r' % (name, kwargs[name], value)
        mesh = getattr(getattr(self.norm, 'args', None), 'mesh_device', None)
        chips = getattr(mesh, 'get_num_devices', None)
        if not callable(chips) or chips() != CENSUS_CHIPS:
            return 'the mesh is not %d chips' % CENSUS_CHIPS
        try:
            mesh_shape = tuple(mesh.shape)
        except Exception:
            mesh_shape = None
        if mesh_shape != CENSUS_MESH_SHAPE:
            return 'the mesh shape %r is not %r' % (mesh_shape, CENSUS_MESH_SHAPE)
        return None

    def __call__(self, tensor, *args, **kwargs):
        plan = self.plan
        reason = self.refusal(tensor, args, kwargs)
        if reason is not None:
            ccl_options_tp.note_fallback(plan, 'ag', reason)
            return self.gather(tensor, *args, **kwargs)
        plan.ag_engaged += 1
        call = dict(kwargs)
        call.update(ccl_options_tp.ag_overrides(plan, self.real))
        if plan.audit_open('ag'):
            return self.audited(tensor, call, kwargs)
        return self.gather(tensor, **call)

    def audited(self, tensor, call, kwargs):
        """The gather with the set's options and with the model's own values, each held in DRAM for audit_round; the model's own values' result is
        served. One gather more than an unaudited call (the plan's quota is even, so the cycling of the semaphore pair stays in step)."""
        real = self.real
        mine = self.gather(tensor, **call)
        mine_view = tile_collective_tp._view_of(mine)
        copy_mine = tile_collective_tp.dram_copy(real, mine)
        real.deallocate(mine)
        tt_ccl = self.norm.tt_ccl
        own = dict(kwargs)
        own['multi_device_global_semaphore'] = tt_ccl.get_and_cycle_ag_semaphore_handles()
        own['barrier_semaphore'] = tt_ccl.get_and_cycle_barrier_semaphore_handle()
        try:
            served = self.gather(tensor, **own)
        except BaseException:
            real.deallocate(copy_mine)
            raise
        shape = tuple(served.shape)
        tile_collective_tp.hold_pair(real, copy_mine, served, (shape[2], shape[3]), mine_view, 'ag',
                                     (ccl_options_tp.AUDIT_MARKER, ccl_options_tp.AUDIT_MISMATCH_MARKER))
        return served


def scoped_forward(original):
    """DistributedNorm.forward inside the block scope's gather shim; `original` is the pinned method, called unchanged."""
    namespace = original.__globals__

    @functools.wraps(original)
    def forward(self, x, *args, **kwargs):
        plan = ccl_options_tp.current()
        if plan is None or getattr(self, 'TG', False) or not _is_decode(args, kwargs):
            return original(self, x, *args, **kwargs)
        real = namespace['ttnn']
        shim = GatherShim(plan, self, real)
        view = _View(real, experimental=_View(real.experimental, all_gather_async=shim))
        namespace['ttnn'] = view
        try:
            return original(self, x, *args, **kwargs)
        finally:
            namespace['ttnn'] = real

    forward.gather_scope_of = original
    return forward


def _is_decode(args, kwargs):
    mode = kwargs.get('mode', args[0] if args else None)
    return getattr(mode, 'value', mode) == 'decode'


def install(module=None, check_source=True):
    """Wrap DistributedNorm.forward. -> [(namespace, name, original)] for tp_addresses to put back; [] when QWEN_FAST_CCL_OPTIONS is unset, when the
    model tree is not importable here (the CPU suite) or when it is already wrapped. A forward whose source is not the pinned one raises."""
    if ccl_options_tp.settings() is None:
        return []
    if module is None:
        try:
            module = importlib.import_module(NORM_MODULE)
        except ImportError as error:
            if getattr(error, 'name', None) in (NORM_MODULE.split('.')[0], NORM_MODULE, 'models.tt_transformers', 'models.tt_transformers.tt'):
                return []
            raise
    cls = getattr(module, CLASS_NAME, None)
    if cls is None:
        return []
    current = cls.forward
    if getattr(current, 'gather_scope_of', None) is not None:
        return []
    if check_source and forward_digest(current) != FORWARD_SHA256:
        raise RuntimeError('%s.forward is not the pinned source (sha256 %s, pinned %s): the gather options are proven against that text only'
                           % (CLASS_NAME, forward_digest(current), FORWARD_SHA256))
    cls.forward = scoped_forward(current)
    return [(tile_collective_tp.ClassAttributes(cls), 'forward', current)]
