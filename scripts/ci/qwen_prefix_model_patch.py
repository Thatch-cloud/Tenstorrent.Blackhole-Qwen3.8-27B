"""G1 model graft: stage conversation prefix reuse into the image's model.py and qwen36_vllm.py.

The model side of the TT prefix-reuse design (revision 2, 2026-09-26): section 2.0.1 item 4 and
the "Model graft" row of section 2.2. vLLM keeps the attention KV pages and the scheduler graft
(prefix_scheduler_graft / qwen_prefix_registry) trims every hit to Q, a 2048-token boundary with a
saved GatedDeltaNet (GDN) checkpoint, and commits a grant per admitted request. This stage makes
the model honour a grant exactly:

model.py (models/demos/blackhole/qwen36/tt/model.py in the image)
  * both chunk loops, _prefill_traced_chunked_tp and _prefill_chunked_eager_tp, become resumable
    through lever_n_model_patch.patch_tp_replay (chunk_from, chunk_to, do_reset, do_tail; Lever N
    M1's edit, reused as the design says), and gain a capture hook (capture_at, on_capture) that
    runs once a chunk ending on a planned boundary has run, before any later chunk or the tail;
  * the eager loop's tail guard (F1): a resumed row whose new tokens stay inside one chunk runs no
    full chunk, and the stock tail would call ttnn.deallocate(None);
  * prefill_traced_chunked gains start / capture_at / on_capture and passes them to the loops only
    when prefix reuse engages. RoPE stays staged for the whole prompt on every call: Lever N's
    "build RoPE only at start == 0" edit targets the same anchor and is NOT adopted, and a tree
    carrying it is refused (F5);
  * prefill_paged_slots_prefix, the per-row prefill for the QWEN_PREFIX_REUSE route: every row is
    checked against its committed grant before any row runs (grant.q == start_pos, the grant is the
    row's, the checkpoint's token ids equal the prompt's; F2), a hit restores fp32 rec_state and
    conv_carry into the bound B=1 scratch in place and resumes at Q/2048, planned boundaries are
    captured to host (a failure skips the checkpoint and is counted, never fails the request; S7),
    and QWEN_PREFIX_AUDIT=1 logs program-free KV / GDN-slot / logits digests (F3);
  * the restore path is chosen and compiled at warmup, before any trace is parked (F3):
    copy_host_to_device_tensor (no allocation, no program) when a per-chip pattern round-trips
    exactly, else from_torch + ttnn.copy, whose programs the warmup has already compiled.

qwen36_vllm.py (models/demos/blackhole/qwen36/tt/qwen36_vllm.py)
  * supports_prefix_caching = QWEN_PREFIX_REUSE == "1", read once at import. The flag and the graft
    ship in this one stage: with the flag on and no graft, the stock model ignores start_pos and
    silently rewrites shared blocks;
  * the batched TP prefill routes to prefill_paged_slots_prefix with the runner's start_pos and the
    row request ids (REQ_IDS_KWARG, supplied by the runner patch); a resumed row reaching the
    unbatched path is an assertion;
  * warmup_model_prefill warms the restore path on its first call (the plugin's compile-only phase).

With QWEN_PREFIX_REUSE unset every served path is the stock one: the new keyword arguments default
to the stock behaviour, prefill_paged_slots is untouched, and the capability evaluates to False.
test_qwen_prefix_model_runtime executes both files against a recording fake ttnn and holds that.

Anchor checks: the stage refuses unless the two files' sha256 equal SOURCE_SHA256 (the IMG tree,
md5 e4ba08d9 / b5230935; the image's copy is UNVERIFIED to match until the bring-up anchor probe
reads it), and refuses to write unless the result equals PATCHED_SHA256 - so a drifted
lever_n_model_patch (the bundle's 77d6995a copy has no eager-loop edit) cannot stage a different
graft. Every edit is also scoped to one method and must match exactly once.

Usage (in the image, after the C2 model-tree graft):
    python3 qwen_prefix_model_patch.py --tree /opt/tt-metal/models/demos/blackhole/qwen36/tt
    python3 qwen_prefix_model_patch.py --tree ... --probe       # original / staged / unknown
Applying nothing on import: stage() is explicit, like serving_plugin_patch and lever_n_model_patch.
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import lever_n_model_patch
from lever_n_model_patch import function_span, replace_once

MODEL_FILE = 'model.py'
VLLM_FILE = 'qwen36_vllm.py'
SOURCE_SHA256 = {
    MODEL_FILE: 'c977f3808c39c9dacde5a62a1e30c09dbb55b27d272fecaa9ffea09991270391',
    VLLM_FILE: 'cda38c3121b7a61417885469c224c0c69189fda899fbf8361565f4d93125c2fe',
}
# The graft.sha256 half: what the stage must produce from the pinned originals. Any change to an
# edit below, or to lever_n_model_patch.patch_tp_replay, changes these on purpose
# (test_qwen_prefix_model_patch prints the new values).
PATCHED_SHA256 = {
    MODEL_FILE: 'e5414fb9d06680cf213399e4ee78d95ed4dcd0d814009acaf016ce21cbf772b8',
    VLLM_FILE: 'e59106fbc15d00e8f399803a36c65d9ae16f667a6b8624d20687d2086694455f',
}

# Shared with the scheduler graft (prefix_scheduler_graft.REGISTRY_KEY) and the runner patch.
REGISTRY_KEY = '_qwen_prefix_registry'
REQ_IDS_KWARG = 'qwen_prefix_req_ids'
CHUNK = 2048
MARKER_ROW = '[PREFIX] row='
MARKER_WARM = '[PINDIAG] prefix: model warm restore_mode='
MARKER_AUDIT = '[PREFIX-AUDIT]'
STAGED_SIGN_MODEL = '_QWEN_PREFIX_REGISTRY_KEY'
STAGED_SIGN_VLLM = '_QWEN_PREFIX_REUSE'

TP_FUNCTION = lever_n_model_patch.TP_FUNCTION
EAGER_FUNCTION = lever_n_model_patch.EAGER_FUNCTION
CHUNKED_FUNCTION = lever_n_model_patch.CHUNKED_FUNCTION
SLOT_WRITE_FUNCTION = '_write_gdn_slot'

# Lever N trees are refused (F5): its entry edit builds RoPE only at start == 0 on the same anchor
# this stage keeps, and under Lever N start_pos > 0 also means "continue my own suspended scratch".
LEVER_N_SIGNS_MODEL = (
    ('def prefill_paged_slots_range(', 'Lever N M1 prefill_paged_slots_range'),
    ('        if start == 0:\n            self._build_request_rope(', "Lever N M1's RoPE-at-start-0 entry edit"),
    ('chunk_from=0, chunk_to=None, do_reset=True, do_tail=True', 'resumable chunk loops (Lever N M1)'),
    ('resumable prefill must continue on a chunk boundary', "Lever N M1's chunked entry"),
)
LEVER_N_SIGNS_VLLM = (
    ('[M1] prefill path', "Lever N M1's vLLM entry"),
    ('prefill_paged_slots_range', 'Lever N M1 prefill_paged_slots_range'),
)

IMPORT_ANCHOR = 'from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq\n'

MODEL_ADAPTER = r'''
# ---- Prefix reuse (G1): the model side of the TT prefix-reuse design, section 2.0.1 item 4. ----
# Staged by scripts/ci/qwen_prefix_model_patch.py. Only qwen36_vllm's QWEN_PREFIX_REUSE route
# (Qwen36Model.prefill_paged_slots_prefix) reaches this code; the stock methods take the new
# keyword arguments at defaults that reproduce their old behaviour exactly.
import time as _qwen_time

_QWEN_PREFIX_REGISTRY_KEY = "_qwen_prefix_registry"
_QWEN_PREFIX_CHUNK = 2048


def _qwen_prefix_registry():
    """The scheduler graft's registry, parked under a fixed sys.modules key (None when absent).

    The adapter. The model reads only: registry.grant_for(req_id) -> None or a grant with req_id,
    q (0 on a miss), checkpoint (pos, token_ids, rec, carry; required when q > 0) and plan (the
    boundaries to capture, as (pos, key) pairs or bare positions); registry.capture(req_id, pos,
    rec=, carry=, nbytes=), which must not raise; and the optional registry.stats dict. If the
    registry's API moves, change these module functions, not the model methods."""
    import sys as _qwen_sys

    return getattr(_qwen_sys.modules.get(_QWEN_PREFIX_REGISTRY_KEY), "registry", None)


def _qwen_prefix_note(registry, name, value):
    stats = getattr(registry, "stats", None)
    if isinstance(stats, dict) and name in stats:
        stats[name] += value


def _qwen_prefix_put(registry, req_id, pos, rec, carry, nbytes):
    try:
        return registry.capture(req_id, pos, rec=rec, carry=carry, nbytes=nbytes)
    except Exception as error:  # the registry guards itself; a capture never fails a request (S7)
        _qwen_prefix_note(registry, "capture_failures", 1)
        logger.warning(f"[PREFIX] capture not stored req={req_id} pos={pos}: {error!r}")
        return None


def _qwen_prefix_plan(grant):
    positions = set()
    for item in getattr(grant, "plan", None) or ():
        positions.add(int(item[0] if isinstance(item, (tuple, list)) else item))
    return sorted(positions)


class _QwenPrefixRow:
    """One prefill row's reuse decision, checked before any row of the step runs."""

    __slots__ = ("index", "req_id", "start", "actual", "rec", "carry", "plan", "dropped")

    def __init__(self, index, req_id, start, actual, rec=None, carry=None, plan=(), dropped=()):
        self.index = index
        self.req_id = req_id
        self.start = start
        self.actual = actual
        self.rec = rec
        self.carry = carry
        self.plan = list(plan)
        self.dropped = list(dropped)


def _qwen_prefix_row(index, req_id, start, actual, toks, registry, chunk_size):
    """Check one row against its committed grant (F2) and return what the model runs.

    The scheduler graft commits a grant only for a request the step's output admits at
    start_pos == Q, so every refusal here is a broken invariant, not a miss: the engine stops
    rather than rewrite KV blocks other conversations share or resume from another state."""
    start = int(start)
    where = f"prefix reuse: row {index} req={req_id} start_pos={start} L={actual}"
    if not 0 <= start < actual:
        raise AssertionError(f"{where}: start_pos is outside the prompt")
    grant = registry.grant_for(req_id) if registry is not None and req_id is not None else None
    if grant is None:
        if start:
            if registry is None:
                reason = "no prefix registry is installed"
            elif req_id is None:
                reason = "the runner passed no request id"
            else:
                reason = "no committed grant"
            raise AssertionError(f"{where}: resumed with {reason}; refusing to rewrite shared blocks")
        return _QwenPrefixRow(index, req_id, 0, actual)
    if getattr(grant, "req_id", req_id) != req_id:
        raise AssertionError(f"{where}: the committed grant belongs to {grant.req_id}")
    q = int(grant.q)
    if q != start:
        raise AssertionError(f"{where}: stale grant Q={q}; the runner resumed at start_pos")
    rec = carry = None
    if q:
        if q % chunk_size:
            raise AssertionError(f"{where}: Q={q} is not a {chunk_size}-token chunk boundary")
        checkpoint = getattr(grant, "checkpoint", None)
        if checkpoint is None or int(checkpoint.pos) != q:
            raise AssertionError(f"{where}: the grant carries no checkpoint at Q={q}")
        if list(checkpoint.token_ids) != toks[0, :q].tolist():
            raise AssertionError(f"{where}: the checkpoint's token ids differ from the row's prompt")
        rec, carry = checkpoint.rec, checkpoint.carry
        if rec is None or carry is None:
            raise AssertionError(f"{where}: the checkpoint holds no GDN state")
    last = actual // chunk_size * chunk_size
    plan, dropped = [], []
    for pos in _qwen_prefix_plan(grant):
        (plan if q < pos <= last and pos % chunk_size == 0 else dropped).append(pos)
    return _QwenPrefixRow(index, req_id, q, actual, rec, carry, plan, dropped)


def _qwen_prefix_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
'''

MODEL_METHODS = r'''
    def prefill_paged_slots_prefix(
        self, token_ids_list, page_table, empty_slots, valid_lens=None, starts=None, req_ids=None
    ):
        """prefill_paged_slots with conversation prefix reuse (QWEN_PREFIX_REUSE=1; G1).

        Rows run in sequence into the persistent B=1 scratch exactly as prefill_paged_slots runs
        them, except:
          * a row granted a hit at Q (== the runner's start_pos) restores its GDN checkpoint
            (rec_state + conv_carry) into the bound scratch in place and runs the chunk loop from
            Q/2048, then the same tail; KV [0, Q) is vLLM's cached blocks. Same chunk program, same
            inputs from Q on, so KV [Q, L), the slot and the logits equal a cold prefill's (design
            section 2.0.4);
          * at each planned 2048-token boundary the loop hands the scratch's state to the registry,
            which keys it by vLLM's block hash at that boundary.
        Every row is checked before any row runs (_qwen_prefix_row): a stale or missing grant is an
        assertion, never a silent rewrite of blocks another conversation shares.
        """
        assert self.num_devices > 1, "prefill_paged_slots_prefix is the TP (num_devices>1) path"
        N = len(token_ids_list)
        assert len(empty_slots) == N, "one slot per request"
        starts = [0] * N if starts is None else [int(s) for s in starts]
        req_ids = [None] * N if req_ids is None else list(req_ids)
        assert len(starts) == N and len(req_ids) == N, "one start_pos and one request id per row"
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt.shape[0] == N, "page_table must have one row per request"
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = self._qwen_prefix_gdn_layers()
        registry = _qwen_prefix_registry()
        chunk_size = self._chunked_chunk_size or _QWEN_PREFIX_CHUNK
        path = "traced" if self._chunked_trace_id is not None else "eager"
        rows = []
        for u in range(N):
            toks = token_ids_list[u]
            assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
            actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
            assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
            rows.append(_qwen_prefix_row(u, req_ids[u], starts[u], actual, toks, registry, chunk_size))
        audit = os.environ.get("QWEN_PREFIX_AUDIT") == "1"

        prev = self._bind_gdn_prefill_scratch()
        host_logits = []
        per_user_rec = []
        per_user_conv = []
        try:
            for row in rows:
                u, actual = row.index, row.actual
                toks = token_ids_list[u]
                began = _qwen_time.perf_counter()
                restored_ms = None
                programs = None
                if row.start:
                    before = self._qwen_prefix_program_cache_entries()
                    self._qwen_prefix_restore(row.rec, row.carry)
                    restored_ms = (_qwen_time.perf_counter() - began) * 1000.0
                    programs = (before, self._qwen_prefix_program_cache_entries())
                    _qwen_prefix_note(registry, "restore_ms", restored_ms)
                captured = []
                resume = {}
                if row.start or row.plan:
                    resume["start"] = row.start
                if row.plan:
                    resume["capture_at"] = frozenset(row.plan)
                    resume["on_capture"] = lambda pos, _row=row, _out=captured: self._qwen_prefix_capture(
                        registry, _row.req_id, pos, _out
                    )
                lg = self.prefill_traced_chunked(toks[:, :actual], pt[u : u + 1], actual_len=actual, **resume)
                host_logits.append(
                    ttnn.to_torch(lg, mesh_composer=comp).reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)
                )
                ttnn.deallocate(lg)
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
                )
                if audit:
                    self._qwen_prefix_audit(row, pt[u : u + 1], per_user_rec[-1], per_user_conv[-1], host_logits[-1])
                elapsed = (_qwen_time.perf_counter() - began) * 1000.0
                restored = "-" if restored_ms is None else f"{restored_ms:.1f}"
                logger.info(
                    f"[PREFIX] row={u} req={row.req_id} path={path} Q={row.start} L={actual} "
                    f"restored_ms={restored} captured=[{','.join(captured)}] dropped={row.dropped} "
                    f"ms={elapsed:.1f} programs_across_restore={programs}"
                )
        finally:
            # Always rebind the batched decode buffers; the scratch persists (see prefill_paged_slots).
            self._unbind_gdn_prefill_scratch(prev)

        for u in range(N):
            self._write_gdn_slot(int(empty_slots[u]), per_user_rec[u], per_user_conv[u])
        return host_logits

    def _qwen_prefix_gdn_layers(self):
        return [layer.attention for layer in self.layers if not layer.is_full_attention]

    def _qwen_prefix_program_cache_entries(self):
        """Program-cache size, for the F3 check that a restore compiles nothing (None if unknown)."""
        try:
            return int(self.mesh_device.num_program_cache_entries())
        except Exception:
            return None

    def _qwen_prefix_read_scratch(self):
        """Host copies of the bound scratch's GDN state, per GDN layer: rec_state
        [num_devices, Nv, Dk, Dv] (fp32 unless QWEN35_GDN_STATE_BF16=1) and conv_carry
        [num_devices, K-1, qkv_dim_tp]. The same blocking D2H every prefill runs for its slot."""
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        rec, carry, nbytes = [], [], 0
        for dn in self._qwen_prefix_gdn_layers():
            r = ttnn.to_torch(dn.rec_state, mesh_composer=comp)
            c = ttnn.to_torch(dn.conv_carry, mesh_composer=comp)
            rec.append(r)
            carry.append(c)
            nbytes += r.element_size() * r.nelement() + c.element_size() * c.nelement()
        return rec, carry, nbytes

    def _qwen_prefix_capture(self, registry, req_id, pos, captured):
        """Capture hook: a chunk loop calls it once the chunk ending at pos has run and the device
        has synchronized, before any later chunk or the tail. A failure (MemoryError included)
        skips the checkpoint and is counted; it never fails the request (S7)."""
        began = _qwen_time.perf_counter()
        try:
            rec, carry, nbytes = self._qwen_prefix_read_scratch()
        except Exception as error:
            _qwen_prefix_note(registry, "capture_failures", 1)
            logger.warning(f"[PREFIX] capture skipped req={req_id} pos={pos}: {error!r}")
            captured.append(f"{pos}:skipped")
            return
        stored = _qwen_prefix_put(registry, req_id, pos, rec, carry, nbytes)
        ms = (_qwen_time.perf_counter() - began) * 1000.0
        _qwen_prefix_note(registry, "capture_ms", ms)
        captured.append(f"{pos}:{'stored' if stored is not None else 'refused'}:{ms:.0f}ms")

    def _qwen_prefix_restore(self, rec_list, carry_list, mode=None):
        """Write a checkpoint into the bound B=1 prefill scratch IN PLACE, so the addresses the
        chunk trace baked stay valid. rec_state and conv_carry only: every resumed chunk and the
        tail rewrite all K conv_states (capture_state=True on prefill).

        h2d:  ttnn.copy_host_to_device_tensor, the traced loop's own input path - it allocates
              nothing and runs no program.
        copy: from_torch + ttnn.copy (_restore_gdn_scratch's pattern plus the carry); its programs
              are compiled by _qwen_prefix_warm_restore before any trace is parked.
        A restore before the warmup chose a path is refused: a first compile after parking is the
        second-request hang (#48536), not a wrong answer."""
        mode = mode or getattr(self, "_qwen_prefix_restore_mode", None)
        if mode not in ("h2d", "copy"):
            raise AssertionError(
                "prefix reuse: GDN restore before _qwen_prefix_warm_restore chose a path "
                "(a first compile after the traces are parked hangs the engine)"
            )
        layers = self._qwen_prefix_gdn_layers()
        if len(rec_list) != len(layers) or len(carry_list) != len(layers):
            raise AssertionError(
                f"prefix reuse: checkpoint holds {len(rec_list)}/{len(carry_list)} GDN states, the model {len(layers)}"
            )
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        held = []
        for dn, rec, carry in zip(layers, rec_list, carry_list):
            for host, target in ((rec, dn.rec_state), (carry, dn.conv_carry)):
                if mode == "h2d":
                    src = ttnn.from_torch(
                        host, dtype=target.dtype, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=mapper
                    )
                    ttnn.copy_host_to_device_tensor(src, target)
                    held.append(src)  # read by the queued write until the device synchronizes
                else:
                    src = ttnn.from_torch(
                        host, dtype=target.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mapper
                    )
                    ttnn.copy(src, target)
                    ttnn.deallocate(src)
        ttnn.synchronize_device(self.device)
        held.clear()

    def _qwen_prefix_warm_restore(self):
        """Choose the GDN restore path and compile it before any trace is parked (F3).

        Called with the B=1 scratch bound, from Qwen36ForCausalLM.warmup_model_prefill's first
        call (the plugin's compile-only phase). Each path writes a per-chip pattern and reads it
        back. copy runs first and always, so that falling back to it can never be a first compile
        after parking; h2d is preferred when it round-trips exactly. QWEN_PREFIX_RESTORE=h2d|copy
        forces one. The engine refuses to start when no path round-trips."""
        rec_now, carry_now, nbytes = self._qwen_prefix_read_scratch()

        def pattern(t):
            base = (torch.arange(t.numel()) % 64).to(torch.float32).reshape(t.shape) / 8.0
            chip = torch.arange(t.shape[0], dtype=torch.float32).reshape([-1] + [1] * (t.dim() - 1))
            return (base + chip * 0.5).to(t.dtype)

        rec_pat = [pattern(t) for t in rec_now]
        carry_pat = [pattern(t) for t in carry_now]
        results = {}
        for mode in ("copy", "h2d"):
            try:
                self._qwen_prefix_restore(rec_pat, carry_pat, mode=mode)
                rec_back, carry_back, _ = self._qwen_prefix_read_scratch()
                exact = all(torch.equal(a, b) for a, b in zip(rec_back + carry_back, rec_pat + carry_pat))
                results[mode] = "exact" if exact else "differs"
            except Exception as error:
                results[mode] = f"refused ({type(error).__name__}: {error})"
        self._reset_gdn_state_for_new_sequence()
        forced = os.environ.get("QWEN_PREFIX_RESTORE", "")
        order = (forced,) if forced in ("h2d", "copy") else ("h2d", "copy")
        chosen = next((mode for mode in order if results.get(mode) == "exact"), None)
        if chosen is None:
            raise RuntimeError(f"prefix reuse: no GDN restore path round-trips the prefill scratch: {results}")
        self._qwen_prefix_restore_mode = chosen
        kv = self._paged_kv_caches[0][0].dtype if self._paged_kv_caches else None
        logger.info(
            f"[PINDIAG] prefix: model warm restore_mode={chosen} results={results} "
            f"gdn_layers={len(rec_now)} checkpoint_bytes={nbytes} "
            f"rec={tuple(rec_now[0].shape)}/{rec_now[0].dtype} carry={tuple(carry_now[0].shape)}/{carry_now[0].dtype} "
            f"kv_dtype={kv} programs={self._qwen_prefix_program_cache_entries()}"
        )

    def _qwen_prefix_audit(self, row, page_row, rec_snap, conv_snap, logits):
        """QWEN_PREFIX_AUDIT=1: digests the exactness gate compares between a hit and a salted cold
        run of the same prompt (F3).

        Program-free: each paged KV cache is read to host with to_torch, ONE tensor at a time (all
        of them at once would need ~34 GB of host RAM), and the row's blocks are selected on host -
        a device-side slice would compile after the traces are parked. The digests are per
        2048-token window of the row's logical sequence, over the unpacked values every reader
        sees, so a hit (new windows from Q/2048) lines up with a cold run (windows from 0). The GDN
        slot snapshot and the logits get one digest each."""
        import hashlib as _qwen_hashlib

        actual = row.actual
        chunk = _QWEN_PREFIX_CHUNK
        block_size = get_block_size(self._paged_kv_caches)
        n_blocks = -(-actual // block_size)
        blocks = torch.as_tensor(page_row[0, :n_blocks], dtype=torch.long)
        windows = [_qwen_hashlib.sha256() for _ in range(-(-actual // chunk))]
        heads = ttnn.ConcatMeshToTensor(self.mesh_device, dim=1)
        for k_cache, v_cache in self._paged_kv_caches:
            for cache in (k_cache, v_cache):
                host = ttnn.to_torch(cache, mesh_composer=heads)
                sel = host.index_select(0, blocks)
                del host
                seq = sel.permute(1, 0, 2, 3).reshape(sel.shape[1], n_blocks * block_size, sel.shape[3])
                for w, digest in enumerate(windows):
                    digest.update(_qwen_prefix_bytes(seq[:, w * chunk : min((w + 1) * chunk, actual)]))
                del sel, seq
        for w, digest in enumerate(windows):
            logger.info(
                f"[PREFIX-AUDIT] req={row.req_id} Q={row.start} L={actual} window={w} "
                f"tokens=[{w * chunk},{min((w + 1) * chunk, actual)}) new={int(w * chunk >= row.start)} "
                f"kv={digest.hexdigest()[:32]}"
            )
        gdn = _qwen_hashlib.sha256()
        for rec in rec_snap:
            gdn.update(_qwen_prefix_bytes(rec))
        for convs in conv_snap:
            for c in convs:
                gdn.update(_qwen_prefix_bytes(c))
        logger.info(
            f"[PREFIX-AUDIT] req={row.req_id} Q={row.start} L={actual} gdn_slot={gdn.hexdigest()[:32]} "
            f"logits={_qwen_hashlib.sha256(_qwen_prefix_bytes(logits)).hexdigest()[:32]}"
        )
'''

LOOP_SIGNATURE_OLD = 'chunk_from=0, chunk_to=None, do_reset=True, do_tail=True'
LOOP_SIGNATURE_NEW = ('chunk_from=0, chunk_to=None, do_reset=True, do_tail=True,\n'
                      '        capture_at=None, on_capture=None')

TRACED_CAPTURE_ANCHOR = (
    '            if (c + 1) % _log_every == 0:\n'
    '                logger.info(f"[TP chunk-replay] {c + 1}/{num_full} chunks")\n')
TRACED_CAPTURE_NEW = TRACED_CAPTURE_ANCHOR + (
    '            # Prefix reuse (G1): GDN state at a planned 2048-token boundary, before any later\n'
    '            # chunk or the tail. A gap boundary inside the loop costs this one extra sync.\n'
    '            if capture_at and (c + 1) * chunk_size in capture_at:\n'
    '                ttnn.synchronize_device(self.device)\n'
    '                _host_refs.clear()\n'
    '                on_capture((c + 1) * chunk_size)\n')

EAGER_TAIL_OLD = (
    '            ttnn.synchronize_device(self.device)\n'
    '        if do_tail and tail_real > 0:\n'
    '            ttnn.deallocate(last_hidden)\n')
EAGER_TAIL_NEW = (
    '            ttnn.synchronize_device(self.device)\n'
    '            # Prefix reuse (G1): GDN state at a planned 2048-token boundary, before the tail.\n'
    '            if capture_at and (c + 1) * chunk_size in capture_at:\n'
    '                on_capture((c + 1) * chunk_size)\n'
    '        if do_tail and tail_real > 0:\n'
    '            # Prefix reuse (F1): a resumed row whose new tokens stay inside one chunk runs no\n'
    '            # full chunk here, so last_hidden is still None (a stock prefill always runs one).\n'
    '            if last_hidden is not None:\n'
    '                ttnn.deallocate(last_hidden)\n')

ENTRY_SIGNATURE_OLD = 'def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None):'
ENTRY_SIGNATURE_NEW = (
    'def prefill_traced_chunked(\n'
    '        self, token_ids, page_table, actual_len, vision_tokens=None, start=0, capture_at=None, on_capture=None\n'
    '    ):')
ENTRY_ROPE = '        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)\n'
ENTRY_RESUME = ENTRY_ROPE + (
    '\n'
    '        # Prefix reuse (G1): resume on a 2048-token boundary with the GDN state already restored\n'
    '        # into the scratch, and capture it at planned boundaries. The RoPE table above stays\n'
    '        # staged for the whole prompt on every call - a resumed row needs exactly that table.\n'
    '        # Nothing below changes a call that passes neither start nor capture_at.\n'
    '        _qwen_resume = {}\n'
    '        if start or capture_at:\n'
    '            assert start % chunk_size == 0 and 0 <= start < actual_len, (\n'
    '                f"prefix reuse: start_pos {start} is not a chunk boundary inside the prompt (L={actual_len})"\n'
    '            )\n'
    '            assert num_full >= 1 and self.num_devices > 1, (\n'
    '                f"prefix reuse needs the chunked TP path: start={start} L={actual_len} num_full={num_full}"\n'
    '            )\n'
    '            assert not capture_at or on_capture is not None, "prefix reuse: capture_at without on_capture"\n'
    '            _qwen_resume = dict(\n'
    '                chunk_from=start // chunk_size,\n'
    '                chunk_to=num_full,\n'
    '                do_reset=(start == 0),\n'
    '                capture_at=capture_at,\n'
    '                on_capture=on_capture,\n'
    '            )\n')
ENTRY_TRACED_OLD = (
    '                return self._prefill_traced_chunked_tp(\n'
    '                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=vision_tokens\n'
    '                )')
ENTRY_TRACED_NEW = (
    '                return self._prefill_traced_chunked_tp(\n'
    '                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=vision_tokens,\n'
    '                    **_qwen_resume,\n'
    '                )')
ENTRY_EAGER_OLD = (
    '                flex_sdpa=True,\n'
    '                vision_tokens=vision_tokens,\n'
    '            )')
ENTRY_EAGER_NEW = (
    '                flex_sdpa=True,\n'
    '                vision_tokens=vision_tokens,\n'
    '                **_qwen_resume,\n'
    '            )')

VLLM_CONSTANTS_ANCHOR = '_BLOCK_SIZE = 64\n'
VLLM_CONSTANTS = (
    '\n'
    '# Prefix reuse (G1, TT prefix-reuse design section 2.0.1 item 4; staged by\n'
    '# scripts/ci/qwen_prefix_model_patch.py). Read once, at import: the platform reads\n'
    '# supports_prefix_caching in check_and_update_config, so the API server and the engine core must\n'
    '# agree, and the serving contract sets profile env in every python process of the image.\n'
    '_QWEN_PREFIX_REUSE = os.environ.get("QWEN_PREFIX_REUSE") == "1"\n'
    "# The runner patch's submit_prefill kwarg: each prefill row's vLLM request id, in row order.\n"
    '_QWEN_PREFIX_REQ_IDS_KWARG = "' + REQ_IDS_KWARG + '"\n')
VLLM_CAPABILITY_OLD = '        "supports_prefix_caching": False,\n'
VLLM_CAPABILITY_NEW = ('        # G1: on only with QWEN_PREFIX_REUSE=1, where the model owns GDN checkpoints at 2048-token\n'
                       '        # boundaries and resumes exactly; the flag and that graft ship in one stage.\n'
                       '        "supports_prefix_caching": _QWEN_PREFIX_REUSE,\n')
VLLM_ENTRY_OLD = (
    '        model = self.model[0]\n'
    '        if model.num_devices > 1 and model.args.max_batch_size > 1:\n')
VLLM_ENTRY_NEW = (
    '        model = self.model[0]\n'
    '        if _QWEN_PREFIX_REUSE and not (model.num_devices > 1 and model.args.max_batch_size > 1):\n'
    '            # Only the batched TP path resumes; the others ignore start_pos and would rewrite\n'
    '            # KV blocks other conversations share.\n'
    '            _qwen_starts = kwargs.get("start_pos")\n'
    '            if _qwen_starts is not None and any(int(s) > 0 for s in _qwen_starts):\n'
    '                raise AssertionError(\n'
    '                    "prefix reuse: a resumed prefill reached the unbatched path "\n'
    '                    f"(num_devices={model.num_devices}, max_batch_size={model.args.max_batch_size})"\n'
    '                )\n'
    '        if model.num_devices > 1 and model.args.max_batch_size > 1:\n')
VLLM_BATCHED_CALL_OLD = (
    '            return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))\n')
VLLM_BATCHED_CALL_NEW = (
    '            if _QWEN_PREFIX_REUSE:\n'
    '                return self._prefill_forward_tp_batched(\n'
    '                    model,\n'
    '                    tokens,\n'
    '                    page_table,\n'
    '                    prompt_lens,\n'
    '                    kwargs.get("empty_slots"),\n'
    '                    start_pos=kwargs.get("start_pos"),\n'
    '                    req_ids=kwargs.get(_QWEN_PREFIX_REQ_IDS_KWARG),\n'
    '                )\n'
    + VLLM_BATCHED_CALL_OLD)
VLLM_BATCHED_SIGNATURE_OLD = 'def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots):'
VLLM_BATCHED_SIGNATURE_NEW = (
    'def _prefill_forward_tp_batched(\n'
    '        self, model, tokens, page_table, prompt_lens, empty_slots, start_pos=None, req_ids=None\n'
    '    ):')
VLLM_SLOTS_OLD = '        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)\n'
VLLM_SLOTS_NEW = (
    '        if _QWEN_PREFIX_REUSE:\n'
    '            # start_pos is the runner\'s num_computed_tokens: Q on a committed hit, else 0.\n'
    '            host_logits = model.prefill_paged_slots_prefix(\n'
    '                token_ids_list, pt, empty_slots, valid_lens=plens, starts=start_pos, req_ids=req_ids\n'
    '            )\n'
    '        else:\n'
    '            host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)\n')
VLLM_WARM_OLD = (
    '        if not enable_trace:\n'
    '            return\n')
VLLM_WARM_NEW = (
    '        if _QWEN_PREFIX_REUSE:\n'
    '            self._qwen_prefix_warm()\n'
    + VLLM_WARM_OLD)
VLLM_WARM_METHOD = r'''
    def _qwen_prefix_warm(self):
        """QWEN_PREFIX_REUSE: choose and compile the GDN restore path before any trace is parked (F3).

        Runs on the first warmup_model_prefill call, which the plugin makes in its compile-only
        phase (enable_trace=False, before any trace capture), so it precedes the chunk trace on the
        traced path and the decode trace on both. Once per process."""
        if getattr(self, "_qwen_prefix_warmed", False):
            return
        self._qwen_prefix_warmed = True
        model = self.model[0]
        if not (model.num_devices > 1 and model.args.max_batch_size > 1):
            logger.warning(
                "[PINDIAG] prefix: model warm skipped - not the batched TP path "
                f"(num_devices={model.num_devices}, max_batch_size={model.args.max_batch_size}); "
                "a resumed row will assert"
            )
            return
        prev = model._bind_gdn_prefill_scratch()
        try:
            model._qwen_prefix_warm_restore()
        finally:
            model._unbind_gdn_prefill_scratch(prev)
'''


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _span_replace(source, function, old, new, what):
    lines = source.splitlines(keepends=True)
    return ''.join(replace_once(lines, function_span(source, function), old, new, what))


def _insert_after_method(source, function, block):
    start, end = function_span(source, function)
    lines = source.splitlines(keepends=True)
    lines[end:end] = block.lstrip('\n').splitlines(keepends=True) + ['\n']
    return ''.join(lines)


def refuse_model(source):
    """The model.py this stage must not touch: a Lever N tree, or one it already staged."""
    if STAGED_SIGN_MODEL in source:
        raise ValueError('model.py already carries the prefix-reuse graft')
    for sign, what in LEVER_N_SIGNS_MODEL:
        if sign in source:
            raise ValueError('model.py carries %s; prefix reuse refuses a Lever N tree (F5)' % what)


def refuse_vllm(source):
    if STAGED_SIGN_VLLM in source:
        raise ValueError('qwen36_vllm.py already carries the prefix-reuse graft')
    for sign, what in LEVER_N_SIGNS_VLLM:
        if sign in source:
            raise ValueError('qwen36_vllm.py carries %s; prefix reuse refuses a Lever N tree (F5)' % what)


def patch_model_source(source):
    """Every model.py edit, each scoped to one method (or the import block) and matched once."""
    refuse_model(source)
    if not hasattr(lever_n_model_patch, '_patch_replay_loop'):
        # The serving bundle's copy (77d6995a) edits only the traced loop: the eager loop would
        # replay chunk zero on a continuation (run 35693338281). Overlay the repository's copy.
        raise ValueError('lever_n_model_patch has no eager-loop edit: the bundle copy, not HEAD')
    if source.count(IMPORT_ANCHOR) != 1:
        raise ValueError('model.py import anchor matched %d times' % source.count(IMPORT_ANCHOR))
    source = source.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + MODEL_ADAPTER, 1)

    source = lever_n_model_patch.patch_tp_replay(source)
    for function in (TP_FUNCTION, EAGER_FUNCTION):
        source = _span_replace(source, function, LOOP_SIGNATURE_OLD, LOOP_SIGNATURE_NEW,
                               '%s capture signature' % function)
    source = _span_replace(source, TP_FUNCTION, TRACED_CAPTURE_ANCHOR, TRACED_CAPTURE_NEW,
                           'traced capture hook')
    source = _span_replace(source, EAGER_FUNCTION, EAGER_TAIL_OLD, EAGER_TAIL_NEW,
                           'eager capture hook and tail guard')

    source = _span_replace(source, CHUNKED_FUNCTION, ENTRY_SIGNATURE_OLD, ENTRY_SIGNATURE_NEW,
                           'chunked entry signature')
    source = _span_replace(source, CHUNKED_FUNCTION, ENTRY_ROPE, ENTRY_RESUME, 'chunked entry resume')
    source = _span_replace(source, CHUNKED_FUNCTION, ENTRY_TRACED_OLD, ENTRY_TRACED_NEW,
                           'chunked entry traced call')
    source = _span_replace(source, CHUNKED_FUNCTION, ENTRY_EAGER_OLD, ENTRY_EAGER_NEW,
                           'chunked entry eager call')

    source = _insert_after_method(source, SLOT_WRITE_FUNCTION, MODEL_METHODS)
    ast.parse(source)
    return source


def patch_vllm_source(source):
    """Every qwen36_vllm.py edit: the capability flag, the routing and the warmup, in one stage."""
    refuse_vllm(source)
    for old, what in ((VLLM_CONSTANTS_ANCHOR, 'constants anchor'), (VLLM_CAPABILITY_OLD, 'capability')):
        if source.count(old) != 1:
            raise ValueError('qwen36_vllm.py %s matched %d times' % (what, source.count(old)))
    source = source.replace(VLLM_CONSTANTS_ANCHOR, VLLM_CONSTANTS_ANCHOR + VLLM_CONSTANTS, 1)
    source = source.replace(VLLM_CAPABILITY_OLD, VLLM_CAPABILITY_NEW, 1)
    source = _span_replace(source, 'prefill_forward', VLLM_ENTRY_OLD, VLLM_ENTRY_NEW, 'unbatched guard')
    source = _span_replace(source, 'prefill_forward', VLLM_BATCHED_CALL_OLD, VLLM_BATCHED_CALL_NEW,
                           'batched route')
    source = _span_replace(source, '_prefill_forward_tp_batched', VLLM_BATCHED_SIGNATURE_OLD,
                           VLLM_BATCHED_SIGNATURE_NEW, 'batched signature')
    source = _span_replace(source, '_prefill_forward_tp_batched', VLLM_SLOTS_OLD, VLLM_SLOTS_NEW,
                           'batched prefix call')
    source = _span_replace(source, 'warmup_model_prefill', VLLM_WARM_OLD, VLLM_WARM_NEW, 'warmup hook')
    source = _insert_after_method(source, 'warmup_model_prefill', VLLM_WARM_METHOD)
    ast.parse(source)
    return source


def patch_tree_bytes(model_bytes, vllm_bytes):
    """Both files in one go: the flag and the graft ship together or not at all."""
    model = patch_model_source(model_bytes.decode('utf-8'))
    vllm = patch_vllm_source(vllm_bytes.decode('utf-8'))
    return model.encode('utf-8'), vllm.encode('utf-8')


def probe(tree, graft_sums=None):
    """The bring-up anchor probe: each file's sha256 and whether it is the pinned original, the
    pinned staged graft, or neither. With graft_sums (the C2 image's /opt/qwen-c2/graft.sha256),
    also each grafted model-tree file against it: GRAFT, not IMG, is what the image serves for the
    five grafted files (design F10)."""
    tree = Path(tree)
    report = {}
    for name in (MODEL_FILE, VLLM_FILE):
        digest = sha256_bytes((tree / name).read_bytes())
        state = ('original' if digest == SOURCE_SHA256[name]
                 else 'staged' if digest == PATCHED_SHA256[name] else 'unknown')
        report[name] = {'sha256': digest, 'state': state}
    if graft_sums:
        for line in Path(graft_sums).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            expected, name = line.split(None, 1)
            name = name.strip().lstrip('*')
            if not name.startswith('graft/') or name.endswith('.orig'):
                continue
            path = tree / name[len('graft/'):]
            digest = sha256_bytes(path.read_bytes()) if path.is_file() else None
            report[name[len('graft/'):]] = {
                'sha256': digest, 'state': 'missing' if digest is None else 'grafted' if digest == expected else 'unknown'}
    return report


def stage(tree, output_dir=None):
    """Patch model.py and qwen36_vllm.py under tree (written to output_dir, default in place).

    Refuses unless both inputs are the pinned originals and both outputs the pinned graft; nothing
    is written unless both files pass."""
    tree = Path(tree)
    target = Path(output_dir) if output_dir else tree
    sources = {name: (tree / name).read_bytes() for name in (MODEL_FILE, VLLM_FILE)}
    for name, data in sources.items():
        digest = sha256_bytes(data)
        if digest != SOURCE_SHA256[name]:
            raise ValueError('%s sha256 %s is not the pinned original %s: the anchor probe refuses '
                             'this tree' % (tree / name, digest, SOURCE_SHA256[name]))
    model, vllm = patch_tree_bytes(sources[MODEL_FILE], sources[VLLM_FILE])
    patched = {MODEL_FILE: model, VLLM_FILE: vllm}
    for name, data in patched.items():
        digest = sha256_bytes(data)
        if digest != PATCHED_SHA256[name]:
            raise ValueError('staged %s sha256 %s is not the pinned graft %s: the stage or '
                             'lever_n_model_patch drifted' % (name, digest, PATCHED_SHA256[name]))
    target.mkdir(parents=True, exist_ok=True)
    for name, data in patched.items():
        (target / name).write_bytes(data)
    return {name: sha256_bytes(data) for name, data in patched.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tree', required=True, help='directory holding model.py and qwen36_vllm.py')
    parser.add_argument('--output-dir', help='write here instead of in place')
    parser.add_argument('--probe', action='store_true', help='report the tree state; write nothing')
    parser.add_argument('--expect', choices=('original', 'staged'),
                        help='with --probe: exit 1 unless model.py and qwen36_vllm.py are in this state')
    parser.add_argument('--graft-sums', help='with --probe: also check the files this graft.sha256 names')
    options = parser.parse_args(argv)
    if options.probe:
        report = probe(options.tree, options.graft_sums)
        for name in sorted(report):
            print('%s %s %s' % (report[name]['state'], report[name]['sha256'], name))
        failed = [name for name in (MODEL_FILE, VLLM_FILE)
                  if options.expect and report[name]['state'] != options.expect]
        failed += [name for name in report if name not in (MODEL_FILE, VLLM_FILE)
                   and report[name]['state'] != 'grafted']
        return 1 if failed else 0
    for name, digest in sorted(stage(options.tree, options.output_dir).items()):
        print('staged %s %s' % (digest, name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
