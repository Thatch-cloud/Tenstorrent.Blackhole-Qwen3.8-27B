# SPDX-License-Identifier: Apache-2.0
# Runs in the tt-vllm image with the gdn_decay op grafted in; see
# scripts/tt-run-spec-decode-generate.sh. Needs test_gdn_decode_multi.py, the hidden-retention
# model.py and the MTP-aware weight_mapping.py mounted.
# See docs/speculative-decoding.md §10z.
# Markers like `§10y` refer to numbered entries in the design log this work was
# recorded in. That log is not public; docs/speculative-decoding.md summarises the
# measurements, dead ends and reasoning the markers point at.
"""Qwen3.8 generating text with speculative decoding, on the **traced** serving path.

§10m built the loop but could only run it eagerly, where decode is host-dispatch-bound (4-5 tok/s
against ~18 for real serving) and the compute saving is invisible. §10o/§10x/§10y then measured
verify and decode under trace, but only as *timers* — neither ever compared a token. This joins
the two: the real loop, on the traced path, emitting real text.

    baseline : NGEN traced decode_forward calls at B=1
    spec     : repeat { MTP proposes K drafts; ONE traced decode_forward at B=K+1 verifies them;
                        accept the matching prefix; roll every GDN layer back to it }

Both arms start from the same GDN state and the same token, so the comparison is on one prompt
with one warm cache, and both pay the same host-side logits readback and argmax.

Three things make this a measurement rather than a demo:

* **A rollback gate with a negative control** (`SD_CHECK=1`). After a T-row verify, rolling the
  GDN state back to n=1 must reproduce the state a single B=1 decode leaves; rolling back to n=T
  must not. Both are reported. Without the second number the first proves nothing.
* **A force-reject control** (`SD_FORCE_REJECT=1`). Drafts that can never be accepted degenerate
  the loop to plain decoding, so `mean_accepted` must be exactly 1.000 and the emitted text must
  match the baseline token for token — that is what proves the bookkeeping (positions, KV, state)
  rather than the accept path.
* **Phase timing.** `t_verify`, `t_draft`, `t_commit` and the readback are reported separately,
  because the §10y speedup model counts only `verify/decode` and silently drops the drafting cost.
  If the MTP head eats the win, that has to be visible.

Output identity is *reported*, not asserted — SCORR-14 (§10n). The verify path agrees with
sequential decode to ~1e-5 and matches argmax step by step, but a T-row projection accumulates
differently from T single-row ones, the device is deterministic, and long generations diverge.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn

K_SPEC = int(os.environ.get("SD_K", "2"))
# Collapse the rollback carry from 12 device ops per GDN layer to 2. Valid here and
# NOT in general -- see commit_prefix's `fused_only`.
_FAST_CARRY = os.environ.get("SD_FAST_CARRY") == "1"
# Device-side argmax for the draft loop. Default OFF so the published 0.88x arm reproduces
# byte-for-byte; SD_DRAFT_DEV_ARGMAX=1 is the optimisation under test.
_DRAFT_DEV_ARGMAX = os.environ.get("SD_DRAFT_DEV_ARGMAX") == "1"
_DRAFT_PROFILE = os.environ.get("SD_DRAFT_PROFILE", "0") != "0"
# 2 = sync after enqueue, attributing device execution to `mtp`/`head` instead of `read`
_DRAFT_PROFILE_SYNC = os.environ.get("SD_DRAFT_PROFILE") == "2"
_DRAFT_FAST_READ = os.environ.get("SD_DRAFT_FAST_READ") == "1"
_DRAFT_PROFILE_TWICE = os.environ.get("SD_DRAFT_PROFILE") in ("3", "4")
_DRAFT_PROFILE_TINY = os.environ.get("SD_DRAFT_PROFILE") == "4"
_DRAFT_RM_READ = os.environ.get("SD_DRAFT_RM_READ") == "1"
_rmread_ck = [0]
_rmread_bad = [0]
_fastread_ck = [0]
_dprof = {"emb": 0.0, "rope": 0.0, "mtp": 0.0, "head": 0.0, "read": 0.0,
          "read2": 0.0, "readt": 0.0, "n": 0}
T_TOK = K_SPEC + 1
NGEN = int(os.environ.get("SD_NGEN", "64"))
IMPL = os.environ.get("SD_IMPL", "fused")
BLOCK = 64
NUM_BLOCKS = int(os.environ.get("SD_BLOCKS", "128"))
DO_CHECK = os.environ.get("SD_CHECK", "1") == "1"
FORCE_REJECT = os.environ.get("SD_FORCE_REJECT") == "1"
SAFE_STAGE = os.environ.get("SD_SAFE_STAGE", "1") == "1"
DO_SPEC = os.environ.get("SD_SPEC", "1") == "1"
GATES = int(os.environ.get("SD_GATES", "3"))
GATE_ADV = int(os.environ.get("SD_GATE_ADV", "6"))
ARMS = int(os.environ.get("SD_ARMS", "1"))
PROMPT = os.environ.get("SD_PROMPT") or (
    "The old lighthouse keeper climbed the spiral stairs each evening at dusk, carrying his lamp "
    "and a battered notebook in which he recorded"
)


class StateSnap:
    """One preallocated snapshot of every GDN layer's recurrent + conv state.

    Copy-only after construction: allocating between traced calls would perturb the timing, and
    a rebind would change a buffer address the captured trace has baked in.
    """

    def __init__(self, gdns):
        self.gdns = gdns
        self.rec = [ttnn.clone(g.rec_state, memory_config=ttnn.DRAM_MEMORY_CONFIG) for g in gdns]
        self.conv = [[ttnn.clone(c, memory_config=ttnn.DRAM_MEMORY_CONFIG) for c in g.conv_states]
                     for g in gdns]

    def save(self):
        for i, g in enumerate(self.gdns):
            ttnn.copy(g.rec_state, self.rec[i])
            for m, c in enumerate(g.conv_states):
                ttnn.copy(c, self.conv[i][m])

    def restore(self):
        for i, g in enumerate(self.gdns):
            ttnn.copy(self.rec[i], g.rec_state)
            for m, c in enumerate(g.conv_states):
                ttnn.copy(self.conv[i][m], c)


class MultiSlotStager:
    """`T` snapshots per layer, for the `plain` arm, which stages through an `on_token` hook.

    Per layer, not model-wide: `forward_decode_multi` runs all `T` tokens for layer 0 before
    layer 1, so a model-wide `stage(t)` is written last by the deepest layer at token 0 and the
    slot holds an incoherent cut (§10m).
    """

    def __init__(self, gdns, nslots):
        self.gdns = gdns
        self.idx = {id(g): i for i, g in enumerate(gdns)}
        self.rec = [[ttnn.clone(g.rec_state, memory_config=ttnn.DRAM_MEMORY_CONFIG) for g in gdns]
                    for _ in range(nslots)]
        self.conv = [[[ttnn.clone(c, memory_config=ttnn.DRAM_MEMORY_CONFIG) for c in g.conv_states]
                      for g in gdns] for _ in range(nslots)]

    def stage_layer(self, g, slot):
        i = self.idx[id(g)]
        ttnn.copy(g.rec_state, self.rec[slot][i])
        for m, c in enumerate(g.conv_states):
            ttnn.copy(c, self.conv[slot][i][m])

    def restore(self, slot):
        for i, g in enumerate(self.gdns):
            ttnn.copy(self.rec[slot][i], g.rec_state)
            for m, c in enumerate(g.conv_states):
                ttnn.copy(self.conv[slot][i][m], c)


def main():
    from models.demos.blackhole.qwen36.tests import test_gdn_decode_multi as tgm
    from models.demos.blackhole.qwen36.tests.test_factory import replicate_to_device
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
    from models.tt_transformers.tt.ccl import TT_CCL
    from transformers import AutoConfig, AutoTokenizer

    snap = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*")
    ckpt = snap[0] if snap else os.environ["HF_MODEL"]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, num_command_queues=2,
                                 trace_region_size=1024 * 1024 * 1024)
    try:
        BMAX = 1 << max(0, (T_TOK - 1)).bit_length()
        cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        gen = Qwen36ForCausalLM.initialize_vllm_model(cfg, mesh, max_batch_size=BMAX,
                                                      max_seq_len=1024)
        model = gen.model[0]
        args = gen.args[0] if hasattr(gen, "args") else model.args
        vocab = args.vocab_size
        gdns = [l.attention for l in model.layers if not l.is_full_attention]
        # §10n: with _stable_state False the B<Bmax arm REBINDS rec_state to the width-1 result
        # instead of writing row 0 of the width-Bmax buffer, so the buffer changes identity and
        # width mid-run and every snapshot slot stops matching what it is copied back into.
        for g in gdns:
            g._stable_state = True
        tok = AutoTokenizer.from_pretrained(ckpt)
        ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
        print(f"SD K={K_SPEC} T={T_TOK} bmax={BMAX} ngen={NGEN} impl={IMPL} "
              f"gdn_layers={len(gdns)} prompt_tokens={len(ids)} force_reject={FORCE_REJECT} "
              f"safe_stage={SAFE_STAGE}", flush=True)

        shape = [NUM_BLOCKS, args.n_local_kv_heads, BLOCK, args.head_dim]
        n_attn = sum(1 for l in model.layers if l.is_full_attention)
        kv = gen.allocate_kv_cache(shape, ttnn.bfloat16, n_attn)
        pt_host = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
        assert len(ids) + NGEN + T_TOK + 8 < NUM_BLOCKS * BLOCK, "prompt+generation exceeds KV pool"

        # ---- the one call both arms go through -------------------------------------------------
        def _call(tokens, positions):
            """One traced decode_forward. B==1 is a decode step; B==T_TOK is a verify."""
            n = len(tokens)
            toks = torch.tensor(tokens, dtype=torch.int32).reshape(n, 1)
            sp = torch.tensor(positions, dtype=torch.int32)
            pt = pt_host.repeat(n, 1).contiguous()      # K5: aliased rows, one block pool
            out = gen.decode_forward(tokens=toks, start_pos=sp, page_table=pt, kv_cache=kv,
                                     enable_trace=True, read_from_device=True)
            lg = out[0] if isinstance(out, tuple) else out
            return lg.reshape(-1, vocab)[:n]

        def _replay_nosync(tokens, positions):
            """Traced call with NO host readback -- exactly what §10y's benchmark issued."""
            n = len(tokens)
            toks = torch.tensor(tokens, dtype=torch.int32).reshape(n, 1)
            sp = torch.tensor(positions, dtype=torch.int32)
            pt = pt_host.repeat(n, 1).contiguous()
            return gen.decode_forward(tokens=toks, start_pos=sp, page_table=pt, kv_cache=kv,
                                      enable_trace=True, read_from_device=False)

        def _untraced(tokens, positions):
            n = len(tokens)
            toks = torch.tensor(tokens, dtype=torch.int32).reshape(n, 1)
            sp = torch.tensor(positions, dtype=torch.int32)
            pt = pt_host.repeat(n, 1).contiguous()
            out = gen.decode_forward(tokens=toks, start_pos=sp, page_table=pt, kv_cache=kv,
                                     enable_trace=False, read_from_device=True)
            lg = out[0] if isinstance(out, tuple) else out
            return lg.reshape(-1, vocab)[:n]

        # ---- the multi-token GDN, wired into the traced verify ---------------------------------
        stage_store = {}
        _FUSED = {"fused": True, "plain": False}[IMPL]
        stager = [None]

        def _patched_fused(self, x):
            st = {}
            out = tgm.forward_decode_multi_fused(self, x, T_TOK, stage=st)
            if not SAFE_STAGE:
                stage_store[id(self)] = st
                return out
            # The fused kernel already returns all T intermediate states, so rollback is a slice
            # rather than T*(1+K) device copies -- but under trace those tensors are allocated
            # DURING capture, and commit_prefix reads them from outside it. Copying into buffers
            # allocated before capture removes that assumption; SD_SAFE_STAGE=0 tests whether the
            # assumption in fact holds.
            slot = stage_store.get(id(self))
            if slot is None:
                slot = stage_store[id(self)] = {
                    "states": ttnn.clone(st["states"], memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    "win": ttnn.clone(st["win"], memory_config=ttnn.DRAM_MEMORY_CONFIG),
                }
            else:
                ttnn.copy(st["states"], slot["states"])
                ttnn.copy(st["win"], slot["win"])
            ttnn.deallocate(st["states"])
            ttnn.deallocate(st["win"])
            return out

        def _patched_plain(self, x):
            cb = (lambda t, _g=self: stager[0].stage_layer(_g, t)) if stager[0] else None
            return tgm.forward_decode_multi(self, x, T_TOK, on_token=cb)

        orig_fd = TPGatedDeltaNet.forward_decode
        _patched = _patched_fused if _FUSED else _patched_plain

        def _commit(n):
            if _FUSED:
                for g in gdns:
                    tgm.commit_prefix(g, stage_store[id(g)], n, T_TOK, fused_only=_FAST_CARRY)
            else:
                stager[0].restore(n - 1)

        # Every persistent device allocation happens HERE, before any trace is captured. Trace
        # capture in tt-metal takes intermediates from the reserved trace region, so a later
        # allocation should not be able to disturb a captured trace -- but the width-3 replay
        # does hang after width-1 traffic (see SD PROBE below), and putting the ~250 MB of
        # snapshot buffers in front of both captures removes one explanation for free.
        if _FAST_CARRY:
            tgm.alloc_spec_carry(gdns)
            print(f'SD fast_carry allocated for {len(gdns)} layers', flush=True)
        start = StateSnap(gdns)
        gate_snap = StateSnap(gdns)
        if not _FUSED:
            stager[0] = MultiSlotStager(gdns, T_TOK)

        # ---- warmup, in the order generator_interface.warmup_decode_buckets uses ---------------
        # "Compile every decode width before capturing any bucket trace." Two untraced calls per
        # width, not one: the hidden-retention patch ALLOCATES on the first call at a new shape
        # (ttnn.clone) and COPIES on every later one, and the same is true of the staging buffers
        # below. A single warm therefore leaves `copy` uncompiled, and the capture pass -- which
        # takes the copy branch -- dies on
        #     TT_FATAL: Cannot load new binaries during trace capture.
        # That is exactly how this was first written, and it is the failure the doubled warm buys
        # off. GDN state is zeroed afterwards, and the KV the warm writes at positions 0..T-1 is
        # overwritten by the prompt.
        for _ in range(2):
            _untraced([ids[0]], [0])
        _call([ids[0]], [0])
        # Hold the width-1 hidden buffer: when the retention patch reallocates at width T the old
        # buffer would otherwise lose its last reference and be freed, while the width-1 decode
        # trace still has a copy into that address baked in.
        w1_hidden = getattr(model, "_last_hidden", None)
        assert w1_hidden is not None, "hidden not retained -- QWEN36_LOAD_MTP=1 and patched model.py?"
        print(f"SD warm width=1 ok, hidden={list(w1_hidden.shape)}", flush=True)

        # NOTE the width-T trace is NOT captured here. MEASURED, twice: a width-3 replay that
        # succeeds immediately after its own capture HANGS THE DEVICE once width-1 replays have
        # run in between -- main thread blocked in read_decode_output, both dispatch threads
        # spinning, no progress in 10 minutes, board needs tt-smi -r. `_mark_trace_buffers_
        # corruptible` says as much: with _tt_allow_decode_trace_buffer_reuse the live traces are
        # allowed to overwrite each other's I/O, and the earlier-captured one does not survive it.
        # §10y's benchmark never hit this because it captures the wide trace AFTER all its
        # narrow work and never goes back. This script now does the same: everything at width 1
        # happens first, the width-T trace is captured once at the very end, and width 1 is
        # never replayed again.
        #
        # `reset_state_inplace` rather than a snapshot restore: it copies from preallocated zero
        # buffers, so the state is genuinely clean AND every buffer keeps the address the
        # captured trace has baked in. A rebinding `reset_state()` here would silently detach
        # the trace from the state it writes.
        addrs = [id(g.rec_state) for g in gdns]
        for g in gdns:
            g.reset_state_inplace()
        assert [id(g.rec_state) for g in gdns] == addrs, "reset_state_inplace rebound rec_state"

        # ---- prompt, ingested one token at a time through the decode path ----------------------
        # Deliberately NOT prefill: §10n constraint 2 says the chunk kernels and the decode kernels
        # are two numerically different realisations of the same model, and mixing them would put
        # an unrelated ~9% argmax disagreement inside the measurement. Feeding the prompt through
        # decode keeps every token on one path, at the cost of len(ids) extra steps of setup.
        t0 = time.perf_counter()
        cur = None
        for i, t in enumerate(ids):
            cur = int(_call([t], [i]).argmax(dim=-1)[0])
        pos = len(ids)
        print(f"SD prompt ingested in {time.perf_counter()-t0:.1f}s, first_token={cur} "
              f"{tok.decode([cur])!r} pos={pos}", flush=True)

        start.save()
        start_tok, start_pos = cur, pos

        # ---- baseline: the traced production decode path ---------------------------------------
        base_out, c, p = [], start_tok, start_pos
        step_t = []
        t0 = time.perf_counter()
        while len(base_out) < NGEN:
            ts = time.perf_counter()
            c = int(_call([c], [p]).argmax(dim=-1)[0])
            step_t.append(time.perf_counter() - ts)
            p += 1
            base_out.append(c)
        t_base = time.perf_counter() - t0
        step_t.sort()
        print(f"SD baseline {NGEN} tokens in {t_base:.3f}s = {NGEN/t_base:.3f} tok/s "
              f"({t_base/NGEN*1e3:.3f} ms/step, median {step_t[NGEN//2]*1e3:.3f}, "
              f"min {step_t[0]*1e3:.3f})", flush=True)
        print(f"SD base_text={tok.decode(base_out)!r}", flush=True)

        # Is the reference reproducible at all? If a traced decode cannot reproduce itself token
        # for token from the same restored state, no divergence downstream means anything.
        start.restore()
        base2, c2, p2 = [], start_tok, start_pos
        t0 = time.perf_counter()
        while len(base2) < NGEN:
            c2 = int(_call([c2], [p2]).argmax(dim=-1)[0])
            p2 += 1
            base2.append(c2)
        t_base2 = time.perf_counter() - t0
        print(f"SD baseline round2 {NGEN} tokens in {t_base2:.3f}s = {NGEN/t_base2:.3f} tok/s "
              f"({t_base2/NGEN*1e3:.3f} ms/step)", flush=True)
        if base2 == base_out:
            print("SD baseline_reproducible=True", flush=True)
        else:
            d = next(i for i in range(NGEN) if base2[i] != base_out[i])
            print(f"SD baseline_reproducible=False first_diff_at={d} "
                  f"run1={base_out[d]} run2={base2[d]}", flush=True)

        if not DO_SPEC:
            return 0

        # The sequential reference for the rollback gate, taken here because it is the LAST
        # width-1 work this process may do.
        p0 = start_pos
        seq = []
        if DO_CHECK:
            start.restore()
            c, p = start_tok, p0
            for _ in range(T_TOK + 1):
                c = int(_call([c], [p]).argmax(dim=-1)[0])
                p += 1
                seq.append(c)
            print(f"SD CHECK sequential {T_TOK + 1} tokens from pos {p0}: {seq}", flush=True)

        # ---- the MTP proposer ------------------------------------------------------------------
        sys.path.insert(0, "/wrap")
        from mtp_module import Qwen36MTP, load_mtp_weights
        from safetensors import safe_open

        rep = ttnn.ReplicateTensorToMesh(mesh)
        sd = load_mtp_weights(ckpt)
        mtp = Qwen36MTP(mesh, args, sd, TT_CCL(mesh))
        mtp.allocate_kv((NUM_BLOCKS, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16)
        mtp_pt = ttnn.from_torch(pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=mesh, mesh_mapper=rep)
        wm = json.load(open(Path(ckpt) / "model.safetensors.index.json"))["weight_map"]
        with safe_open(str(Path(ckpt) / wm["model.language_model.embed_tokens.weight"]),
                       framework="pt") as f:
            E = f.get_tensor("model.language_model.embed_tokens.weight")
        print("SD MTP head built", flush=True)
        # 10ad: every flag announces itself, so an A/B can assert the arm actually engaged.
        print(f"SD flags dev_argmax={_DRAFT_DEV_ARGMAX} profile={_DRAFT_PROFILE} fast_read={_DRAFT_FAST_READ}", flush=True)

        def _propose(hidden_tt, token, mtp_pos):
            """K chained MTP drafts from the target's hidden at the accepted position."""
            out, hid_in, tok_in = [], hidden_tt, token
            for d in range(K_SPEC):
                _t = time.perf_counter
                _a = _t()
                emb = E[tok_in].reshape(1, 1, 1, -1).to(torch.bfloat16)
                emb_tt = replicate_to_device(mesh, emb)
                _b = _t()
                cp = torch.full((1,), mtp_pos + d, dtype=torch.int32)
                cp_tt = ttnn.from_torch(cp, dtype=ttnn.int32, device=mesh, mesh_mapper=rep)
                cos, sin = rot_mats_decode(mesh, args.rope_head_dim, args.max_seq_len,
                                           args.rope_theta, cp)
                _c = _t()
                hid_in = mtp.forward(emb_tt, hid_in, cp_tt, cos, sin, page_table=mtp_pt)
                if _DRAFT_PROFILE_SYNC:
                    ttnn.synchronize_device(mesh)
                _d = _t()
                if _DRAFT_PROFILE:
                    _dprof["emb"] += _b - _a
                    _dprof["rope"] += _c - _b
                    _dprof["mtp"] += _d - _c
                    _dprof["n"] += 1
                # lm_head on device: `h @ W.T` on host is a 248320x5120 fp32 matmul per draft and
                # dominated the whole loop when it was tried that way (§10m).
                # Retention OFF for this call. The MTP shares the TARGET's lm_head, and the
                # retention patch hooks exactly there -- so a draft's own 1-row hidden would
                # overwrite `model._last_hidden`, rebinding it away from the T-row buffer the
                # captured verify trace writes into. The next `_hidden_row(n>1)` then slices a
                # 1-row tensor and dies on
                #     TT_FATAL: Slice start[2] (2) must be less than input tensor shape[2] (1)
                # `_mtp_retain_enabled` re-reads the environment per call, so this is enough;
                # the copy already captured inside the trace is unaffected either way.
                os.environ["QWEN36_LOAD_MTP"] = "0"
                _e = time.perf_counter()
                try:
                    tt_lg = model._lm_head(hid_in)
                finally:
                    os.environ["QWEN36_LOAD_MTP"] = "1"
                if _DRAFT_PROFILE_SYNC:
                    ttnn.synchronize_device(mesh)
                _f = time.perf_counter()
                if _DRAFT_PROFILE:
                    _dprof["head"] += _f - _e
                # `.cpu()` FIRST, then process_output_decode. Handing it a live DEVICE tensor is
                # the same shape of call that hung the device when the rollback gate read
                # rec_state that way; the production readback (`read_decode_output`) always
                # copies to host first.
                _g = time.perf_counter()
                if _DRAFT_DEV_ARGMAX:
                    # Argmax on device: 4 bytes come back instead of the whole vocab row.
                    # `_lm_head` all-gathers, so every device holds the full [1,1,1,vocab] and
                    # one shard's argmax is the answer -- no composer needed. The readback this
                    # replaces is ~1 MB per draft over PCIe x4 behind the C-Payne switch, paid
                    # K times per step, and it is the only host round trip in the draft loop.
                    tt_am = ttnn.argmax(tt_lg, dim=-1)
                    tok_in = int(ttnn.to_torch(ttnn.get_device_tensors(tt_am)[0]).reshape(-1)[0])
                elif _DRAFT_RM_READ and not _rmread_bad[0]:
                    try:
                        _rm = ttnn.to_layout(tt_lg, ttnn.ROW_MAJOR_LAYOUT)
                        row = ttnn.to_torch(ttnn.get_device_tensors(_rm)[0]).reshape(-1)[:vocab]
                        tok_in = int(row.argmax())
                        ttnn.deallocate(_rm)
                    except Exception as _ex:
                        _rmread_bad[0] = 1
                        print(f"SD rm_read UNAVAILABLE, falling back: {_ex}", flush=True)
                        _lgf = model.process_output_decode(tt_lg.cpu(), 1, S=1)
                        tok_in = int(_lgf.reshape(-1, vocab)[0].argmax())
                    if not _rmread_bad[0] and _rmread_ck[0] < 4:
                        _ref = model.process_output_decode(tt_lg.cpu(), 1, S=1)
                        _ref = int(_ref.reshape(-1, vocab)[0].argmax())
                        assert _ref == tok_in, f"rm_read divergence: {tok_in} vs composed {_ref}"
                        _rmread_ck[0] += 1
                        print(f"SD rm_read check {_rmread_ck[0]}/4 ok tok={tok_in}", flush=True)
                elif _DRAFT_FAST_READ:
                    # Shard 0 holds the whole row (the lm_head all-gathers), so the composer is
                    # redundant work on a 248320-wide fp32 tensor.
                    row = ttnn.to_torch(ttnn.get_device_tensors(tt_lg)[0]).reshape(-1)[:vocab]
                    tok_in = int(row.argmax())
                    if _fastread_ck[0] < 4:
                        _ref = model.process_output_decode(tt_lg.cpu(), 1, S=1)
                        _ref = int(_ref.reshape(-1, vocab)[0].argmax())
                        assert _ref == tok_in, f"fast_read divergence: {tok_in} vs composed {_ref}"
                        _fastread_ck[0] += 1
                        print(f"SD fast_read check {_fastread_ck[0]}/4 ok tok={tok_in}", flush=True)
                else:
                    lg = model.process_output_decode(tt_lg.cpu(), 1, S=1)
                    tok_in = int(lg.reshape(-1, vocab)[0].argmax())
                if _DRAFT_PROFILE:
                    _dprof["read"] += time.perf_counter() - _g
                if _DRAFT_PROFILE_TWICE:
                    # Identical readback of an unchanged tensor. Nothing can be pending: the first
                    # read already returned, so this is transfer + host cost with no device wait.
                    _h = time.perf_counter()
                    _lg2 = model.process_output_decode(tt_lg.cpu(), 1, S=1)
                    _t2 = int(_lg2.reshape(-1, vocab)[0].argmax())
                    _dprof["read2"] += time.perf_counter() - _h
                    assert _DRAFT_DEV_ARGMAX or _t2 == tok_in, f"reread differs: {_t2} vs {tok_in}"
                if _DRAFT_PROFILE_TINY:
                    # int32[1]. Same round trip, 1/250000th of the payload.
                    _i = time.perf_counter()
                    _tiny = ttnn.to_torch(ttnn.get_device_tensors(cp_tt)[0]).reshape(-1)[0]
                    _dprof["readt"] += time.perf_counter() - _i
                out.append(tok_in)
            return out

        # ---- STAGE B: width T only, from here to the end ---------------------------------------
        # The width-T trace is captured HERE, after every width-1 replay this process will make
        # and after every persistent allocation (including the MTP head's weights and KV). See
        # the note at the width-1 warm: captured in the other order, this trace dies the moment
        # width 1 is replayed.
        TPGatedDeltaNet.forward_decode = _patched
        probe_toks = [start_tok] + [1] * K_SPEC
        probe_pos = [p0 + i for i in range(T_TOK)]
        start.restore()
        for _ in range(2):
            _untraced(probe_toks, probe_pos)
        print(f"SD warm width={T_TOK} untraced ok, "
              f"staged_layers={len(stage_store) if _FUSED else len(gdns)}", flush=True)
        start.restore()
        _call(probe_toks, probe_pos)
        # Bind the T-row hidden buffer ONCE, here, and read it directly from now on. The trace
        # writes to this address; `model._last_hidden` is only a Python name and anything that
        # calls `_lm_head` outside the trace can move it.
        hid_buf = model._last_hidden
        assert hid_buf.shape[-2] == T_TOK, f"hidden is {list(hid_buf.shape)}, expected {T_TOK} rows"
        print(f"SD warm width={T_TOK} captured, hidden={list(hid_buf.shape)}", flush=True)
        # Probed in two halves so a hang localises itself: an unfinished device program stalls
        # the synchronize, a readback problem stalls only the second call.
        start.restore()
        _replay_nosync(probe_toks, probe_pos)
        ttnn.synchronize_device(mesh)
        print(f"SD warm width={T_TOK} replay(no read) ok", flush=True)
        _commit(1)
        start.restore()
        _call(probe_toks, probe_pos)
        _commit(1)
        print(f"SD warm width={T_TOK} replay(read) ok", flush=True)

        def _vrows(toks, positions):
            return [int(x) for x in _call(toks, positions).argmax(dim=-1)]

        def _hidden_row(n):
            """The target's hidden state at the last ACCEPTED row -- what MTP drafts from."""
            if len(hid_buf.shape) == 4:
                return ttnn.clone(ttnn.slice(hid_buf, (0, 0, n - 1, 0), (1, 1, n, args.dim)))
            return ttnn.clone(ttnn.slice(hid_buf, (0, n - 1, 0), (1, n, args.dim)))

        def _junk():
            return [(i + 1) % vocab for i in range(K_SPEC)]

        # ---- rollback gate, with the control that makes it mean something ----------------------
        # Purely token-level, deliberately, and entirely at width T. The first version read
        # `rec_state` back with `ttnn.to_torch(ttnn.get_device_tensors(...)[0])` on a LIVE DEVICE
        # tensor and hung the device. This one never reads state: it tests the rollback through
        # the only thing that matters, the token the next step produces. "The next token" is
        # itself read as row 0 of a width-T verify -- which item 1 has just shown equals what a
        # width-1 decode produces -- so no width-1 replay is needed after the capture.
        if DO_CHECK:
            # 1. step-wise argmax agreement (§10l), now on the TRACED path -- never checked before
            start.restore()
            got = _vrows([start_tok] + seq[:K_SPEC], probe_pos)
            agree = got == seq[:T_TOK]
            print(f"SD CHECK verify rows={got} expect={seq[:T_TOK]} agree={agree}", flush=True)

            # 2-4. The rollback gate proper, run at several depths. Entirely inside the verify
            # path: comparing against a width-1 decode instead makes the gate hostage to §10n's
            # cross-path disagreement, which is a real ~9%-per-step phenomenon and DID flip the
            # verdict between two otherwise identical runs at position 28 (17199 vs 259 vs 1528,
            # all from the same prefix). Repeating at several depths also stops one near-tie
            # deciding the answer.
            def _run_gate(label, c0, q0):
                gate_snap.save()
                pos_a = [q0 + i for i in range(T_TOK)]
                pos_b = [q0 + 1 + i for i in range(T_TOK)]
                pos_x = [q0 + T_TOK + i for i in range(T_TOK)]
                # The exact continuation, row by row and WITHOUT using rollback: row i of a
                # verify is exact once rows 0..i carry true tokens, and junk further down cannot
                # reach it (later rows write KV only at later positions).
                cont = []
                for i in range(T_TOK):
                    gate_snap.restore()
                    cont.append(_vrows([c0] + cont + _junk()[len(cont):], pos_a)[i])
                # Reference: consume all T true tokens (no rollback at all), then one more step.
                gate_snap.restore()
                _vrows([c0] + cont[:K_SPEC], pos_a)
                X = _vrows([cont[K_SPEC]] + _junk(), pos_x)[0]
                # Arm B -- the rollback the loop actually performs: verify with drafts certain to
                # be rejected, roll back to n=1, then re-consume. Every row must reproduce the
                # reference, the last one landing on the state the reference reached directly.
                gate_snap.restore()
                _vrows([c0] + _junk(), pos_a)
                _commit(1)
                rows_b = _vrows(cont, pos_b)
                # Arm C -- NEGATIVE CONTROL: identical, minus the rollback. The state then holds
                # the junk drafts, so the last row must not match.
                gate_snap.restore()
                _vrows([c0] + _junk(), pos_a)
                _commit(T_TOK)
                rows_c = _vrows(cont, pos_b)

                want = cont[1:] + [X]
                ok_b, ok_c = rows_b == want, rows_c[-1] != X
                print(f"SD GATE {label} pos={q0} cont={cont} X={X}", flush=True)
                print(f"SD GATE {label} rollback rows={rows_b} expect={want} ok={ok_b}",
                      flush=True)
                print(f"SD GATE {label} control  rows={rows_c} last={rows_c[-1]} "
                      f"!= {X} ok={ok_c}", flush=True)
                return ok_b, ok_c

            gate_ok = []
            gc_, gq_ = start_tok, p0
            start.restore()
            for gi in range(GATES):
                gate_ok.append(_run_gate(f"d{gi}", gc_, gq_))
                if gi + 1 == GATES:
                    break
                # Advance to the next depth the way the loop does -- verify, keep row 0, roll
                # back to n=1. If rollback were broken this only moves the gate to a different
                # (still deterministic) state; it cannot make the gate pass.
                gate_snap.restore()
                for _ in range(GATE_ADV):
                    gc_ = _vrows([gc_] + _junk(), [gq_ + i for i in range(T_TOK)])[0]
                    _commit(1)
                    gq_ += 1
            n_b = sum(1 for b, _ in gate_ok if b)
            n_c = sum(1 for _, c_ in gate_ok if c_)
            print(f"SD GATE SUMMARY rollback {n_b}/{GATES} depths reproduce the reference, "
                  f"control separates at {n_c}/{GATES}", flush=True)
            if not agree or n_b < GATES:
                print("SD CHECK FAILED -- rollback did not reproduce the reference", flush=True)
            elif n_c < GATES:
                # Reported, not treated as a pass: at some depths carrying the rejected drafts in
                # the GDN state does not move the argmax, so the gate cannot see a broken rollback
                # there. SD_ARMS=3 is the control with the power the per-step gate lacks.
                print(f"SD CHECK PASSED but UNDER-POWERED at {GATES - n_c}/{GATES} depths -- "
                      f"the no-rollback control did not separate there; rely on SD_ARMS=3",
                      flush=True)
            else:
                print("SD CHECK PASSED at every depth, control separates at every depth",
                      flush=True)

        # ---- speculative -----------------------------------------------------------------------
        def _spec_arm(reject=False, bad_rollback=False):
            """One full speculative generation from `start`. Returns the emitted tokens + stats.

            `reject` replaces the MTP's drafts with valid-but-implausible ids, so nothing can be
            accepted and the loop degenerates to one token per verify. `bad_rollback` always rolls
            back to n=1 whatever was accepted, which is the wrong state whenever n > 1 -- the
            failure the gate is supposed to catch, run at loop scale where it has 60+ steps to
            accumulate instead of one.
            """
            start.restore()
            out, c, p = [], start_tok, start_pos
            st = {"steps": 0, "acc": 0, "commits": 0}
            seen, hit = [0] * K_SPEC, [0] * K_SPEC
            t = {"draft": 0.0, "verify": 0.0, "commit": 0.0, "hidden": 0.0}
            hidden, step_times = None, []
            t_start = time.perf_counter()
            while len(out) < NGEN:
                t_step = time.perf_counter()
                if st["steps"] == 0:
                    # No hidden yet: the first step verifies the single real token to seed it. It
                    # still runs at width T, so it costs a verify -- charged, not excluded.
                    ts = time.perf_counter()
                    got = _vrows([c] + _junk(), [p + i for i in range(T_TOK)])
                    t["verify"] += time.perf_counter() - ts
                    n = 1
                else:
                    ts = time.perf_counter()
                    # MTP position = the index of `c` among GENERATED tokens, the convention the
                    # acceptance harness validated (`cpos = step`, one per emitted token). The
                    # eager §10m loop advanced it by K per STEP instead, so it ran ahead of the
                    # real sequence by every rejected draft and the MTP attended to a KV cache
                    # holding rejected drafts at positions the sequence never occupied. Advancing
                    # per accepted token makes the next step overwrite exactly those entries.
                    drafts = _propose(hidden, c, len(out) - 1)
                    t["draft"] += time.perf_counter() - ts
                    if reject:
                        # Valid-but-implausible ids, NOT negative ones: a negative id reaches the
                        # embedding as a huge uint32, indexes out of range and corrupts the run
                        # rather than controlling it (§10m).
                        drafts = _junk()
                    ts = time.perf_counter()
                    got = _vrows([c] + drafts, [p + i for i in range(T_TOK)])
                    t["verify"] += time.perf_counter() - ts
                    n = 1
                    for i in range(K_SPEC):
                        seen[i] += 1
                        if got[i] == drafts[i]:
                            hit[i] += 1
                            n += 1
                        else:
                            break
                ts = time.perf_counter()
                # No rollback when every draft was accepted: the state the verify left IS the
                # state after n == T tokens. `commit_prefix(g, stage, T, T)` would rewrite it with
                # the same values at a cost of ~720 eager dispatches.
                roll = 1 if bad_rollback else n
                if roll < T_TOK:
                    _commit(roll)
                    st["commits"] += 1
                t["commit"] += time.perf_counter() - ts
                # What is committed is always the VERIFIER's argmax: accepting a draft only means
                # the verifier agreed, so the emitted token is what the target would produce.
                out.extend(got[:n])
                ts = time.perf_counter()
                hidden = _hidden_row(n)
                t["hidden"] += time.perf_counter() - ts
                c, p = out[-1], p + n
                st["steps"] += 1
                st["acc"] += n
                step_times.append(time.perf_counter() - t_step)
            return out, st, seen, hit, t, step_times, time.perf_counter() - t_start

        print("SD spec loop starting", flush=True)
        (spec_out, _st, depth_seen, depth_hit, acc_t,
         spec_step_t, t_spec) = _spec_arm(reject=FORCE_REJECT)
        steps, accepted_total, commits = _st["steps"], _st["acc"], _st["commits"]
        emitted = len(spec_out)
        spec_out_c = spec_out[:NGEN]

        print(f"SD spec {emitted} tokens in {t_spec:.3f}s = {emitted/t_spec:.3f} tok/s "
              f"steps={steps} mean_accepted={accepted_total/steps:.3f}", flush=True)
        print(f"SD SPEEDUP {(emitted/t_spec)/(NGEN/t_base):.3f}x  "
              f"(baseline {NGEN/t_base:.3f} tok/s -> spec {emitted/t_spec:.3f} tok/s)", flush=True)
        per = {k: v / steps * 1e3 for k, v in acc_t.items()}
        if _DRAFT_PROFILE and _dprof['n']:
            _dn = _dprof['n']
            _dsteps = max(1, steps)
            print('SD draft breakdown ms/step: ' + ' '.join(
                f'{k}={_dprof[k] / _dsteps * 1e3:.3f}' for k in
                ('emb', 'rope', 'mtp', 'head', 'read', 'read2', 'readt')) +
                f' | drafts={_dn} steps={_dsteps} per_draft_ms='
                f'{sum(_dprof[k] for k in ("emb","rope","mtp","head","read"))/_dn*1e3:.3f}',
                flush=True)
        print(f"SD phase ms/step: verify={per['verify']:.3f} draft={per['draft']:.3f} "
              f"commit={per['commit']:.3f} hidden={per['hidden']:.3f} "
              f"other={(t_spec/steps*1e3) - sum(per.values()):.3f} "
              f"total={t_spec/steps*1e3:.3f}", flush=True)
        print(f"SD phase ms/token: {t_spec/emitted*1e3:.3f} vs baseline "
              f"{t_base/NGEN*1e3:.3f}  rollbacks={commits}/{steps} "
              f"(commit {acc_t['commit']/max(1, commits)*1e3:.3f} ms each)", flush=True)
        # Median as well as mean. Both arms show occasional outlier steps -- the baseline mean
        # has run 5-10% above its own median between otherwise identical runs -- so a single
        # mean-of-64 ratio is not a reliable instrument at the few-percent level this lands at.
        srt = sorted(spec_step_t)
        med_spec = srt[len(srt) // 2]
        med_tok = med_spec / (accepted_total / steps)
        med_base = step_t[NGEN // 2]
        print(f"SD median ms/step spec={med_spec*1e3:.3f} (min {srt[0]*1e3:.3f}) "
              f"baseline={med_base*1e3:.3f} (min {step_t[0]*1e3:.3f})", flush=True)
        print(f"SD MEDIAN-BASED SPEEDUP {med_base/med_tok:.3f}x "
              f"({med_tok*1e3:.3f} ms/token spec vs {med_base*1e3:.3f} baseline)", flush=True)
        cond = [depth_hit[i] / depth_seen[i] if depth_seen[i] else float("nan")
                for i in range(K_SPEC)]
        print("SD acceptance conditional: " +
              " ".join(f"d{i+1}={cond[i]:.3f}({depth_hit[i]}/{depth_seen[i]})"
                       for i in range(K_SPEC)), flush=True)
        print(f"SD spec_text={tok.decode(spec_out_c)!r}", flush=True)
        same = base_out[:len(spec_out_c)] == spec_out_c
        print(f"SD identical_output={same}", flush=True)
        if not same:
            d = next(i for i in range(len(spec_out_c)) if base_out[i] != spec_out_c[i])
            print(f"SD first_divergence_at={d} base={base_out[d]} spec={spec_out_c[d]} "
                  f"(NOT a failure -- SCORR-14 / §10n)", flush=True)
        pref = 0
        while pref < len(spec_out_c) and base_out[pref] == spec_out_c[pref]:
            pref += 1
        print(f"SD common_prefix={pref}/{NGEN} tokens", flush=True)

        def _cmp(label, other, ref=spec_out_c):
            o = other[:NGEN]
            k = 0
            while k < min(len(o), len(ref)) and o[k] == ref[k]:
                k += 1
            print(f"SD {label} common_prefix_with_arm1={k}/{NGEN} identical={o == ref}",
                  flush=True)
            print(f"SD {label} text={tok.decode(o)!r}", flush=True)
            return k

        # ---- the control that actually tests the loop: change the DRAFTER, keep everything else
        # Speculative decoding's contract is that the drafter cannot change the output -- every
        # emitted token is the verifier's own argmax at a state the rollback restored. So the
        # same start state with a drafter that is never accepted must emit the same tokens. Run
        # in ONE process against ONE start state, because across processes the eager MTP head is
        # not bit-reproducible and the comparison would be confounded.
        if ARMS >= 2:
            rej_out, rst, _s2, _h2, _t2, _p2, t_rej = _spec_arm(reject=not FORCE_REJECT)
            print(f"SD ARM2 reject={not FORCE_REJECT} {len(rej_out)} tokens "
                  f"mean_accepted={rst['acc']/rst['steps']:.3f} in {t_rej:.3f}s "
                  f"= {len(rej_out)/t_rej:.3f} tok/s", flush=True)
            _cmp("ARM2", rej_out)
        # ---- and the control for the rollback itself, at loop scale -----------------------------
        if ARMS >= 3:
            bad_out, bst, _s3, _h3, _t3, _p3, t_bad = _spec_arm(reject=FORCE_REJECT,
                                                                bad_rollback=True)
            print(f"SD ARM3 bad_rollback (always n=1) {len(bad_out)} tokens "
                  f"mean_accepted={bst['acc']/bst['steps']:.3f}", flush=True)
            _cmp("ARM3", bad_out)
        TPGatedDeltaNet.forward_decode = orig_fd
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
