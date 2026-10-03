#!/usr/bin/env python3
"""V5 byte gate on ONE card: the K5-A recurrence launch against the value-split launch (gdn_seq_block_split), bit for bit.

Run with QWEN_FAST_TP=4 and QWEN_FAST_VERIFY_T1=1 in the environment (the width is a launch variable; the coalesced
build is the one the model runs). One p150a, a 1x1 mesh, four users x 16 rows at the four-card geometry (12 value heads,
2,560 channels, 1,536-wide z and output), no model, no collective. Arms:

  A    K5-A, gdn_seq_block.load_kernels(root, 0): the build production runs (its QUALIFIED triple); the reference.
  V    V5, gdn_seq_block_split.load_kernels(root, unqualified=True): the candidate. Only a full-scope PASS here licenses
       committing its sha256 triple (the verdict line's `v5_triple`).
  N5r  negative control: V with the owner's rowsum in the order 2, 3, 0, 1 (what a partial-sum split would give).
  N5x  negative control: V with no exchange and no wait (the owner normalises over stale pages).

Every comparison is bit for bit, on raw page images: ttnn's bf16 upload and readback flush -0.0 and denormals and turn NaN into
-Inf, so every byte crosses as a host-tilized uint32 page image through a raw page copy (the K5-A probe's RAW_COPY, imported
from gdn_seq_block_device_test), and every tensor a launch allocates is first filled with a sentinel page so an unwritten page
cannot pass for an exact one. Compared per user: the gated output (1x16x1536, plus the padded rows 16-31: 48 pages), all 16
snapshots (16x12x128x128: 3,072 pages; snapshot 15 is the carry the commit reads), and that the initial state and every input
are byte-unchanged afterwards.

Sections (--sections, default all; they always run in this order, and cache runs second because its program-cache
deltas only mean something in a process where V has not run yet):
  selftest   a raw round trip of known bytes, and the environment this process read (the launched argv, not the host's)
  cache      program-cache entry deltas: V at 4 users +1, then +0 on fresh addresses with the earlier tensors held (still
             exact); users 1, 2, 3 +1 each; interleaving K5-A does not cross-hit
  p0         every regime, A and V in alternating order: R1 (the gdn_tp4_card_test distributions, five seeds), R2 (wide range),
             R3 (edges: +-0, smallest normal, denormals, zero q/k rows, beta/g extremes), R3b (NaN/Inf poison in the padding),
             RA (column-asymmetric: value columns 64-127 scaled by 2^8, so the norm genuinely mixes the halves), R4 (a
             2,048-token chain of 128 launches, each arm feeding its own states[15] back, compared at every launch), R5
             (real: layer 0's weights and a real coding text through embed, norm, projections, the causal conv and the gate
             parameters, real initial states from a CPU recurrence at offsets 1,024 / 2,048 / 3,072 / 4,080; layers 23 and 47
             as a real-weights proxy on layer-0-style activations)
  users      1, 2 and 3 users (the union of cores, and so the program, differs)
  negative   N5r must differ from A in at least one byte over R1 and RA; N5x, run straight after V on a DIFFERENT input
             set, must differ from A on its own set (stale exchange pages show)
  stale      V(X), V(Y), V(X), each against A on the same set
  trace      A and V captured once each; 100 replays, the inputs restaged in place by raw copy, set X and set Y alternately;
             every V replay against A's replay (semaphore re-initialisation and stale exchange pages)
  timing     48 distinct-input launches per arm in one trace, 25 serpentine rounds: A, V, A-nosnap, V-nosnap and V at depth 1
             (per-launch median and IQR); informs the image-build decision, never the byte verdict

Exit codes: 0 PASS at full scope; 1 FAIL (any differing byte, a blind control, a moved input, an unwritten page, a traced
difference, a wrong cache delta); 2 usage error; 3 the per-call watchdog fired (the partial report is written first; reset the
target card only); 4 NO-DECISION (a section raised, or reduced scope). The last stdout line is one JSON object.

  QWEN_FAST_TP=4 QWEN_FAST_VERIFY_T1=1 python3 gdn_v5_card_m.py --out report.json
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import gdn_multitoken as native
import gdn_seq_block as seq
import gdn_seq_block_device_test as dev
import gdn_seq_block_split as v5
import gdn_tp4_card_test as card_test
import tp_shapes
import verify_trace_t1

VERDICT = 'GDN_V5'
KIND = 'gdn-v5-probe'
# cache runs second: its program-cache deltas are only meaningful in a process where V (and K5-A at four users) has not run yet.
SECTIONS = ('selftest', 'cache', 'p0', 'users', 'negative', 'stale', 'trace', 'timing')
REGIMES = ('R1', 'R2', 'R3', 'R3b', 'RA', 'R4', 'R5')
ROWS = seq.ROWS
HEADS_PER_CHIP = 12
TILE_COLUMNS_PER_HALF = 2
PLAN_R1_SEEDS = (17, 23, 29, 31, 37)
PLAN_OTHER_SEEDS = (17,)
PLAN_USERS = 4
PLAN_R4_LAUNCHES = 128
PLAN_R5_LAYERS = (0, 23, 47)
PLAN_TRACE_REPLAYS = 100
PLAN_TIMING_LAUNCHES = 48
PLAN_TIMING_ROUNDS = 25
R5_OFFSETS = (1024, 2048, 3072, 4080)
RA_SCALE = 2.0 ** 8
# The timing labels (e, anchored on K5-A's measured 193.4 us a layer): what V - A must clear.
PROCEED_US, IMAGE_BUILD_US, WRITE_BOUND_US = -65.0, -40.0, 15.0
NOSNAP = 'nosnap'
ARM_BUILDS = ('A', 'V', 'N5r', 'N5x')
TIMING_ARMS = ('A', 'V', 'A-nosnap', 'V-nosnap', 'V-depth1')
MODULES = ('gdn_seq_block.py', 'gdn_seq_block_compute.cpp', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp',
           'gdn_seq_block_split.py', 'gdn_seq_block_split_compute.cpp', 'gdn_seq_block_split_reader.cpp',
           'gdn_seq_block_split_writer.cpp', 'gdn_seq_block_device_test.py', 'gdn_tp4_card_test.py',
           'gdn_user_batch.py', 'gdn_user_batch_tp.py', 'gdn_multitoken.py', 'tp_shapes.py', 'verify_trace_t1.py',
           'gdn_v5_card_m.py')
ENV_READ = ('QWEN_FAST_TP', 'QWEN_FAST_VERIFY_T1', 'QWEN_FAST_VERIFY_T1_SKIP', 'QWEN_FAST_GDN_SEQ_BLOCK',
            'QWEN_FAST_GDN_SPLIT_V', 'TT_METAL_HOME', 'TT_METAL_CACHE', 'HF_MODEL', 'TT_METAL_WATCHER')


# ---- geometry-aware locators and the pure verdicts (no ttnn: held by test_gdn_v5_card_m on CPU) ----

def locate_output(index, width=1536):
    """The element of the (rows, width) output image a flat index names: its token row (or a padded row), head, tile."""
    row, column = divmod(index, width)
    return dict(token=row if row < ROWS else None, padding_row=row >= ROWS, head=column // 128,
                tile=(column % 128) // 32, half='owner' if (column % 128) // 32 < TILE_COLUMNS_PER_HALF else 'helper',
                element=[row % 32, column % 32])


def locate_states(index, heads=HEADS_PER_CHIP):
    """The element of the (16, heads, 128, 128) snapshot image a flat index names: token, head, K tile i, V tile j and
    the half that held it (j // 2: owner or helper), element."""
    token, rest = divmod(index, heads * 128 * 128)
    head, rest = divmod(rest, 128 * 128)
    row, column = divmod(rest, 128)
    value_tile = column // 32
    return dict(token=token, head=head, tile_i=row // 32, tile_j=value_tile,
                half='owner' if value_tile < TILE_COLUMNS_PER_HALF else 'helper', element=[row % 32, column % 32])


def host_inputs(torch, regime, seed, users, found):
    """(norm_w, [dict(qkv, beta, gate, initial, z) per user], poison per user) as bf16, at the four-card geometry."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    nv, qkv_width, z_width, key = found.gdn_nv, found.gdn_qkv, found.gdn_z, found.gdn_key
    norm_w = (1 + randn(1, 1, 128) * 0.1).bfloat16()
    groups, poison = [], []
    for user in range(users):
        if regime in ('R1', 'R3b', 'R4', 'RA'):
            qkv, beta, gate, initial, z = card_test.user_inputs(torch, found, generator)
            qkv, beta, gate, initial, z = (value.float() for value in (qkv, beta, gate, initial, z))
            if regime == 'RA':
                # value columns 64-127 of every head 2^8 larger than columns 0-63, in v and in the state: the norm's
                # sum of squares is then dominated by the helper's columns, so a reordering shows.
                v = qkv[..., 2 * key:].reshape(1, ROWS, nv, 128)
                v[..., 64:] *= RA_SCALE
                qkv[..., 2 * key:] = v.reshape(1, ROWS, nv * 128)
                initial[..., 64:] *= RA_SCALE
        elif regime == 'R2':
            initial = randn(1, nv, 128, 128) * 4
            qkv, beta, gate, z = randn(1, ROWS, qkv_width) * 8, rand(1, ROWS, nv), -rand(1, ROWS, nv) * 20, \
                randn(1, ROWS, z_width)
        elif regime == 'R3':
            initial = randn(1, nv, 128, 128) * 0.05
            flat = initial.view(-1)
            specials = (0.0, -0.0, 2.0 ** -126, -2.0 ** -126, 2.0 ** -130, -2.0 ** -133)
            chosen = torch.randperm(flat.numel(), generator=generator)[:len(specials) * 1024]
            for index, value in enumerate(specials):
                flat[chosen[index * 1024:(index + 1) * 1024]] = value
            qkv, z = randn(1, ROWS, qkv_width), randn(1, ROWS, z_width)
            qkv[0, 3, key:2 * key] = 0.0   # an all-zero k row: the k-norm eps path
            qkv[0, 7, 0:key] = 0.0         # and an all-zero q row: the q-norm eps path
            beta = torch.where(rand(1, ROWS, nv) < 0.5, torch.tensor(0.0), torch.tensor(1 - 2.0 ** -8))
            gate = torch.where(rand(1, ROWS, nv) < 0.5, torch.tensor(0.0), torch.tensor(-88.0))
        else:
            raise ValueError('Unknown regime %r' % (regime,))
        groups.append(dict(qkv=qkv.bfloat16(), beta=beta.bfloat16(), gate=gate.bfloat16(),
                           initial=initial.bfloat16(), z=z.bfloat16()))
        poison.append((float('nan'), float('nan'), float('inf'), float('-inf'))[user % 4] if regime == 'R3b' else None)
    return norm_w, groups, poison


def timing_label(a_us, v_us, v_nosnap_us=None):
    """The timing verdict from per-launch medians: V - A <= -65 us proceeds (-3.1 ms a four-user verify), <= -40 us
    is an image build, anything else is kill-and-zone-profile; `write_bound` when V - V_nosnap >= 15 us (the snapshot
    writes, not the compute, set the time)."""
    if a_us is None or v_us is None:
        return dict(label='not-run', delta_us=None, write_bound=None)
    delta = v_us - a_us
    label = 'proceed' if delta <= PROCEED_US else 'image-build' if delta <= IMAGE_BUILD_US else 'kill'
    write_bound = None if v_nosnap_us is None else (v_us - v_nosnap_us) >= WRITE_BOUND_US
    return dict(label=label, delta_us=delta, verify_ms_est=48 * delta / 1000, write_bound=write_bound)


def scope_missing(arguments):
    """What a run lacks of the full gate: empty means scope=full."""
    missing = ['section %s' % name for name in SECTIONS if name not in arguments.sections]
    missing += ['regime %s' % name for name in REGIMES if name not in arguments.regimes]
    if 'R1' in arguments.regimes:
        missing += ['R1 seed %d' % seed for seed in PLAN_R1_SEEDS if seed not in arguments.seeds]
    if 'R4' in arguments.regimes and arguments.r4_launches < PLAN_R4_LAUNCHES:
        missing.append('R4 launches %d of %d' % (arguments.r4_launches, PLAN_R4_LAUNCHES))
    if 'R5' in arguments.regimes:
        missing += ['R5 layer %d' % layer for layer in PLAN_R5_LAYERS if layer not in arguments.r5_layers]
    if arguments.users != PLAN_USERS:
        missing.append('users %d of %d' % (arguments.users, PLAN_USERS))
    if 'trace' in arguments.sections and arguments.trace_replays < PLAN_TRACE_REPLAYS:
        missing.append('trace replays %d of %d' % (arguments.trace_replays, PLAN_TRACE_REPLAYS))
    if 'timing' in arguments.sections and (arguments.timing_launches < PLAN_TIMING_LAUNCHES
                                           or arguments.timing_rounds < PLAN_TIMING_ROUNDS):
        missing.append('timing %d launches x %d rounds (plan %d x %d)' % (
            arguments.timing_launches, arguments.timing_rounds, PLAN_TIMING_LAUNCHES, PLAN_TIMING_ROUNDS))
    return missing


def decide(report):
    """(verdict, problems) from the report: FAIL on any hard miss, NO-DECISION when something raised or the scope is
    reduced, PASS only when nothing is missing at full scope. A hard miss outranks an error: a differing byte is a
    finding even if a later section raised."""
    fail, undecided = [], []
    sections = report.get('sections', {})
    for name, section in sections.items():
        if section.get('error'):
            undecided.append('section %s raised: %s' % (name, section['error']))
    if not report.get('a_qualified', False):
        undecided.append('control A is not the qualified K5-A build (the byte gate is against production)')
    tallies = report.get('tallies', {})
    if 'V' in tallies and tallies['V']['cases'] and not dev.arm_exact(tallies['V']):
        first = tallies['V'].get('first_failure') or {}
        fail.append('V differs from A: %d cases, %d exact, %d differing bytes (first: %s)' % (
            tallies['V']['cases'], tallies['V']['exact_cases'], tallies['V']['differing_bytes'], first.get('case')))
    elif 'V' not in tallies or not tallies['V']['cases']:
        undecided.append('no V case ran')
    negative = sections.get('negative', {})
    if negative and not negative.get('error'):
        for control in ('N5r', 'N5x'):
            if not negative.get(control, {}).get('detects'):
                fail.append('negative control %s did not differ from A: the compare is blind to it' % control)
    if report.get('inputs_moved'):
        fail.append('an input moved: %s' % report['inputs_moved'][:3])
    if report.get('unwritten'):
        fail.append('pages never written: %s' % report['unwritten'][:3])
    stale = sections.get('stale', {})
    if stale and not stale.get('error') and not stale.get('exact'):
        fail.append('stale-input sequence V(X) V(Y) V(X) differs from A')
    trace = sections.get('trace', {})
    if trace and not trace.get('error') and not (trace.get('exact') and not trace.get('unwritten')):
        fail.append('traced replays differ from A')
    cache = sections.get('cache', {})
    if cache and not cache.get('error') and not cache.get('ok'):
        fail.append('program cache: %s' % cache.get('problems'))
    if report.get('missing_scope'):
        undecided.append('reduced scope: %s' % '; '.join(report['missing_scope'][:6]))
    if fail:
        return 'FAIL', fail + undecided
    if undecided:
        return 'NO-DECISION', undecided
    return 'PASS', []


def exit_code(verdict):
    return {'PASS': 0, 'FAIL': 1, 'NO-DECISION': 4}[verdict]


def verdict_line(report):
    v = report.get('tallies', {}).get('V') or {}
    timing = report.get('sections', {}).get('timing', {})
    return '%s verdict=%s scope=%s p0=%s/%s bytes=%s traced=%s timing=%s a_us=%s v_us=%s' % (
        VERDICT, report['verdict'], 'reduced' if report.get('missing_scope') else 'full',
        v.get('exact_cases', 0), v.get('cases', 0), v.get('differing_bytes', 'n/a'),
        'not-run' if 'trace' not in report.get('sections', {}) else
        ('exact' if report['sections']['trace'].get('exact') and not report['sections']['trace'].get('unwritten')
         else 'error' if report['sections']['trace'].get('error') else 'differs'),
        timing.get('label', 'not-run'), _us(timing.get('a_us')), _us(timing.get('v_us')))


def _us(value):
    return 'n/a' if value is None else '%.1f' % value


# ---- real inputs (R5): layer weights and a real coding text, on CPU ----

def snapshot_directory(model_dir):
    """The Hugging Face snapshot directory under `model_dir` (itself, or its only snapshots/<rev>)."""
    path = Path(model_dir)
    if (path / 'config.json').exists():
        return path
    found = sorted((path / 'snapshots').glob('*/config.json'))
    if not found:
        raise FileNotFoundError('no snapshot with a config.json under %s' % model_dir)
    return found[-1].parent


class Weights:
    """Read named tensors (rows or whole) out of a sharded safetensors snapshot, one open file at a time."""

    def __init__(self, snapshot):
        from safetensors import safe_open
        self._open = safe_open
        self.snapshot = Path(snapshot)
        index = self.snapshot / 'model.safetensors.index.json'
        if index.exists():
            self.map = json.loads(index.read_text())['weight_map']
        else:
            with safe_open(str(self.snapshot / 'model.safetensors'), 'pt') as handle:
                self.map = {name: 'model.safetensors' for name in handle.keys()}

    def find(self, suffix):
        names = [name for name in self.map if name.endswith(suffix)]
        if len(names) != 1:
            raise KeyError('expected one tensor ending %r, found %d' % (suffix, len(names)))
        return names[0]

    def shape(self, name):
        with self._open(str(self.snapshot / self.map[name]), 'pt') as handle:
            return list(handle.get_slice(name).get_shape())

    def slice(self, name, start=None, stop=None):
        with self._open(str(self.snapshot / self.map[name]), 'pt') as handle:
            handle_slice = handle.get_slice(name)
            return handle_slice[start:stop] if start is not None else handle.get_tensor(name)

    def rows(self, name, ids):
        with self._open(str(self.snapshot / self.map[name]), 'pt') as handle:
            handle_slice = handle.get_slice(name)
            return [handle_slice[i:i + 1][0] for i in ids]

    def gdn_layers(self):
        """Model layer indices that carry a linear_attn block, ascending."""
        found = []
        for name in self.map:
            marker = '.layers.'
            if marker in name and name.endswith('.linear_attn.in_proj_qkv.weight'):
                found.append(int(name.split(marker)[1].split('.')[0]))
        return sorted(found)


def causal_conv_silu(torch, x, taps):
    """x [T, C] float, taps [C, K]: out[t, c] = silu(sum_j taps[c, j] * x[t - (K-1) + j, c]), zero history."""
    kernel = taps.shape[1]
    padded = torch.cat([torch.zeros(kernel - 1, x.shape[1], dtype=x.dtype), x], dim=0)
    total = sum(taps[:, j][None, :] * padded[j:j + x.shape[0]] for j in range(kernel))
    return torch.nn.functional.silu(total)


def recurrence_states(torch, qkv, beta, gate, found, offsets):
    """The state before token `offset` for each offset: the sequential fp32 recurrence of one chip's 12 value heads,
    the state rounded to bf16 after every token (as the kernel carries it). Returns {offset: [1, 12, 128, 128] bf16}."""
    nv, key = found.gdn_nv, found.gdn_key
    groups = nv // found.gdn_nk
    q = qkv[:, :key].reshape(-1, found.gdn_nk, 128).repeat_interleave(groups, dim=1)
    k = qkv[:, key:2 * key].reshape(-1, found.gdn_nk, 128).repeat_interleave(groups, dim=1)
    v = qkv[:, 2 * key:].reshape(-1, nv, 128)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    decay = torch.exp(gate)
    state = torch.zeros(nv, 128, 128)
    saved = {}
    for token in range(max(offsets)):
        if token in offsets:
            saved[token] = state.bfloat16()[None].clone()
        state = state * decay[token][:, None, None]
        read = torch.einsum('hk,hkv->hv', k[token], state)
        delta = (v[token] - read) * beta[token][:, None]
        state = (state + k[token][:, :, None] * delta[:, None, :]).bfloat16().float()
    if max(offsets) in offsets:
        saved[max(offsets)] = state.bfloat16()[None].clone()
    return saved


def real_inputs(torch, model_dir, text_path, layer_rank, found, users, offsets=R5_OFFSETS):
    """(norm_w, users_host, poison, meta): GDN layer number `layer_rank` (0 = the first linear_attn layer, 23 and 47 by
    GDN position) of the snapshot, chip 0's slice of it (key heads 0-3, value heads 0-11 of the flat q|k|v channel
    order the checkpoint stores), on a real coding text: embed -> input RMSNorm -> in_proj_qkv/z/a/b -> causal conv +
    silu -> beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias). One user per offset; the 16 rows start there, and
    the initial state is the CPU recurrence's at that offset. For layer_rank > 0 the activations are layer 0's embedding
    run through THIS layer's norm and projections: real weights on layer-0-style activations (a proxy), said so in meta."""
    from transformers import AutoTokenizer
    snapshot = snapshot_directory(model_dir)
    weights = Weights(snapshot)
    layers = weights.gdn_layers()
    if layer_rank >= len(layers):
        raise ValueError('layer rank %d: the snapshot has %d linear_attn layers' % (layer_rank, len(layers)))
    layer = layers[layer_rank]
    prefix = None
    for name in weights.map:
        if name.endswith('.layers.%d.linear_attn.in_proj_qkv.weight' % layer):
            prefix = name[:-len('in_proj_qkv.weight')]
            break
    block = prefix[:-len('linear_attn.')]
    tokens_needed = max(offsets) + ROWS
    text = Path(text_path).read_text(encoding='utf-8', errors='replace')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    ids, repeats = [], 0
    while len(ids) < tokens_needed:
        ids += tokenizer(text, add_special_tokens=False).input_ids
        repeats += 1
    ids = ids[:tokens_needed]
    embed = torch.stack(weights.rows(weights.find('embed_tokens.weight'), ids)).float()
    norm_name = block + 'input_layernorm.weight'
    gamma = weights.slice(norm_name).float()
    offset_one = 1.0  # Qwen3-Next style zero-centred RMSNorm: scaled by (1 + w); recorded, it moves realism only
    normed = embed * torch.rsqrt((embed * embed).mean(-1, keepdim=True) + 1e-6) * (offset_one + gamma)
    normed = normed.bfloat16().float()
    qkv_weight = prefix + 'in_proj_qkv.weight'
    total = weights.shape(qkv_weight)[0]
    value_total = weights.shape(prefix + 'in_proj_z.weight')[0]
    key_total = (total - value_total) // 2
    key, value = found.gdn_key, found.gdn_value
    if 2 * key + value != found.gdn_qkv:
        raise AssertionError('geometry: 2*%d + %d != %d' % (key, value, found.gdn_qkv))
    q_w = weights.slice(qkv_weight, 0, key)
    k_w = weights.slice(qkv_weight, key_total, key_total + key)
    v_w = weights.slice(qkv_weight, 2 * key_total, 2 * key_total + value)
    qkv = normed @ torch.cat([q_w, k_w, v_w], dim=0).float().T
    conv = weights.slice(prefix + 'conv1d.weight').float()[:, 0, :]
    taps = torch.cat([conv[0:key], conv[key_total:key_total + key], conv[2 * key_total:2 * key_total + value]], dim=0)
    qkv = causal_conv_silu(torch, qkv, taps).bfloat16()
    z = (normed @ weights.slice(prefix + 'in_proj_z.weight', 0, found.gdn_z).float().T).bfloat16()
    a = normed @ weights.slice(prefix + 'in_proj_a.weight', 0, found.gdn_nv).float().T
    b = normed @ weights.slice(prefix + 'in_proj_b.weight', 0, found.gdn_nv).float().T
    a_log = weights.slice(prefix + 'A_log', 0, found.gdn_nv).float()
    dt_bias = weights.slice(prefix + 'dt_bias', 0, found.gdn_nv).float()
    beta = torch.sigmoid(b).bfloat16()
    gate = (-torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias)).bfloat16()
    norm_w = weights.slice(prefix + 'norm.weight').float().bfloat16().reshape(1, 1, 128)
    states = recurrence_states(torch, qkv.float(), beta.float(), gate.float(), found, tuple(offsets))
    groups = []
    for user in range(users):
        start = offsets[user]
        rows = slice(start, start + ROWS)
        groups.append(dict(qkv=qkv[rows][None].contiguous(), beta=beta[rows][None].contiguous(),
                           gate=gate[rows][None].contiguous(), initial=states[start], z=z[rows][None].contiguous()))
    meta = dict(layer=layer, gdn_rank=layer_rank, proxy=layer_rank != 0, offsets=list(offsets[:users]),
                tokens=len(ids), text_repeats=repeats, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                rms_offset=offset_one, tensors=prefix)
    return norm_w, groups, [None] * users, meta


# ---- the watchdog (the verify_t2 pattern, minimal: one armed deadline, the partial report, os._exit(3)) ----

class Watchdog:
    def __init__(self, on_fire):
        self.on_fire = on_fire
        self.seconds = 0.0

    @contextmanager
    def guard(self, label, seconds=None):
        seconds = self.seconds if seconds is None else seconds
        if not seconds:
            yield
            return
        timer = threading.Timer(seconds, self.fire, [label])
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def fire(self, label):
        print('WATCHDOG: %r exceeded its deadline; writing the partial report and exiting 3 (the target card may be '
              'hung: reset it alone, never the serving pair without the hint)' % (label,), flush=True)
        try:
            self.on_fire(label)
        finally:
            os._exit(3)


# ---- arguments ----

def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sections', default=','.join(SECTIONS))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in PLAN_R1_SEEDS), help='R1 seeds')
    parser.add_argument('--other-seeds', default=','.join(str(seed) for seed in PLAN_OTHER_SEEDS),
                        help='R2, R3, R3b and RA seeds')
    parser.add_argument('--users', type=int, default=PLAN_USERS, choices=(1, 2, 3, 4))
    parser.add_argument('--r4-launches', type=int, default=PLAN_R4_LAUNCHES)
    parser.add_argument('--r4-seed', type=int, default=41)
    parser.add_argument('--r5-layers', default=','.join(str(layer) for layer in PLAN_R5_LAYERS),
                        help='GDN layer positions (0 = the first linear_attn layer)')
    parser.add_argument('--real-text', type=Path, default=Path('/bench/gdn_seq_block.py'),
                        help='a mounted repo source file, the coding text of R5')
    parser.add_argument('--model-dir', type=Path, default=Path(os.environ.get('HF_MODEL', '')))
    parser.add_argument('--trace-replays', type=int, default=PLAN_TRACE_REPLAYS)
    parser.add_argument('--timing-launches', type=int, default=PLAN_TIMING_LAUNCHES)
    parser.add_argument('--timing-rounds', type=int, default=PLAN_TIMING_ROUNDS)
    parser.add_argument('--timing-arms', default=','.join(TIMING_ARMS))
    parser.add_argument('--call-timeout', type=float, default=180.0, help='per-call watchdog, seconds (0 = off)')
    parser.add_argument('--output-memory', choices=('l1', 'dram'), default='l1')
    parser.add_argument('--trace-region', type=int, default=134217728)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    arguments = parser.parse_args(argv)
    for name in ('sections', 'regimes', 'timing_arms'):
        setattr(arguments, name, [item for item in getattr(arguments, name).split(',') if item])
    arguments.seeds = [int(item) for item in arguments.seeds.split(',') if item]
    arguments.other_seeds = [int(item) for item in arguments.other_seeds.split(',') if item]
    arguments.r5_layers = [int(item) for item in arguments.r5_layers.split(',') if item]
    if any(name not in SECTIONS for name in arguments.sections):
        parser.error('--sections names only %s' % (SECTIONS,))
    if any(name not in REGIMES for name in arguments.regimes):
        parser.error('--regimes names only %s' % (REGIMES,))
    if any(name not in TIMING_ARMS for name in arguments.timing_arms) or 'A' not in arguments.timing_arms:
        parser.error('--timing-arms names only %s and includes A' % (TIMING_ARMS,))
    if arguments.r4_launches < 1 or arguments.trace_replays < 2 or arguments.timing_launches < 1:
        parser.error('--r4-launches, --trace-replays (>= 2) and --timing-launches must be positive')
    return arguments


# ---- the device part ----

def main(argv=None):
    arguments = parse(argv)
    here = Path(__file__).resolve().parent
    ci = Path(dev.__file__).resolve().parent
    report = dict(scope='V5 (gdn_seq_block_split) against K5-A (gdn_seq_block) on one card, four-card geometry; no model, no '
                        'collective', argv=sys.argv[1:], users=arguments.users, rows=ROWS, sections={}, tallies={},
                  stages=[], unwritten=[], inputs_moved=[], a_qualified=False, verdict='NO-DECISION',
                  missing_scope=scope_missing(arguments),
                  env_read={name: os.environ.get(name) for name in ENV_READ},
                  native_sha256=native.HASHES,
                  module_sha256={name: hashlib.sha256((path).read_bytes()).hexdigest()
                                 for name in MODULES for path in (here / name, ci / name) if path.exists()})
    summary = dict(kind=KIND, verdict='NO-DECISION')
    mesh = [None]

    def write(extra=None):
        report.update(extra or {})
        arguments.out.write_text(json.dumps(report, indent=2, default=str))

    watchdog = Watchdog(lambda label: write(dict(error='watchdog: %r' % (label,), watchdog=label)))
    watchdog.seconds = arguments.call_timeout

    def stage(label, **details):
        report['stages'].append(dict(stage=label, **details))
        write()
        print(json.dumps(report['stages'][-1], default=str), flush=True)

    code = 4
    try:
        stage('environment', **report['env_read'])
        import torch
        import ttnn
        report['ttnn_path'] = ttnn.__file__
        if os.environ.get('QWEN_FAST_TP') != '4' or tp_shapes.chip_count() != 4:
            raise AssertionError('QWEN_FAST_TP=4 is required (the launched environment says %r)'
                                 % os.environ.get('QWEN_FAST_TP'))
        if not verify_trace_t1.cut('coalesce'):
            raise AssertionError('QWEN_FAST_VERIFY_T1=1 with the coalesce cut is required: both arms must run as the '
                                 'coalesced build the model runs')
        found = tp_shapes.geometry(4)
        if (found.gdn_nv, found.gdn_qkv, found.gdn_z, found.gdn_value) != (HEADS_PER_CHIP, 2560, 1536, 1536):
            raise AssertionError('four-card geometry drifted: %r' % (found,))

        stage('load-kernels', root=str(arguments.root))
        builds = dict(A=seq.load_kernels(arguments.root, 0, unqualified=True),
                      V=v5.load_kernels(arguments.root, unqualified=True),
                      N5r=v5.load_kernels(arguments.root, variant='N5r', unqualified=True),
                      N5x=v5.load_kernels(arguments.root, variant='N5x', unqualified=True))
        builds['A-nosnap'] = seq.load_kernels(arguments.root, 0, diag=NOSNAP, unqualified=True)
        builds['V-nosnap'] = v5.load_kernels(arguments.root, diag=NOSNAP, unqualified=True)
        builds['V-depth1'] = v5.load_kernels(arguments.root, depth=1, unqualified=True)
        report['a_qualified'] = bool(builds['A'].qualified)
        report['generated_sha256'] = dict(A=seq.sha256(builds['A']), V=v5.sha256(builds['V']))
        report['v5_triple'] = v5.sha256(builds['V'])
        report['cb_bytes_per_core'] = dict(A=seq.cb_bytes(), V=v5.cb_bytes(), V_depth1=v5.cb_bytes(1))
        report['placement'] = [[list(point) for _, _, point in pairs][:4] for pairs in v5.placement(11, 10, arguments.users)]

        stage('mesh-open')
        mesh[0] = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576,
                                        trace_region_size=arguments.trace_region)
        device = mesh[0]
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        output_memory = ttnn.L1_MEMORY_CONFIG if arguments.output_memory == 'l1' else ttnn.DRAM_MEMORY_CONFIG

        def raw_copy(source, destination, pages):
            workers = min(dev.RAW_WORKERS, pages)
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(workers - 1, 0))])
            scratch = ttnn.CBDescriptor(total_size=2048, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048,
                                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
            program = ttnn.MeshProgramDescriptor()
            for chip, (left, right) in enumerate(zip(ttnn.get_device_tensors(source),
                                                     ttnn.get_device_tensors(destination), strict=True)):
                args = []
                for value in (left, right):
                    args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
                runtime = ttnn.RuntimeArgs()
                for worker in range(workers):
                    runtime[worker][0] = [left.buffer_address(), right.buffer_address(), pages, worker, workers]
                kernel = ttnn.KernelDescriptor(kernel_source=dev.RAW_COPY,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
                kernel.runtime_args = runtime
                coordinate = ttnn.MeshCoordinate(0, chip)
                program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
                    kernels=[kernel], cbs=[scratch])
            with watchdog.guard('raw_copy'):
                ttnn.generic_op([source, destination], program)

        def words_tensor(words):
            return ttnn.from_torch(words.reshape(1, 1, -1, 512), device=device, dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(device))

        def sync(label):
            with watchdog.guard(label):
                ttnn.synchronize_device(device)

        sentinel = words_tensor(torch.full((dev.MAX_PAGES, 512), dev.SENTINEL_WORD, dtype=torch.int32))

        def fill_sentinel(value):
            pages = dev.page_count(value.shape)
            if pages > dev.MAX_PAGES:
                raise ValueError('A launch allocated %d pages; the sentinel holds %d' % (pages, dev.MAX_PAGES))
            raw_copy(sentinel, value, pages)

        def upload_exact(logical, image=None):
            image = dev.pad_image(torch, logical) if image is None else image
            if tuple(image.shape) != dev.padded_shape(logical.shape):
                raise ValueError('Image is not the padded shape of the logical tensor')
            target = ttnn.empty(tuple(logical.shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            words = dev.tile_image(torch, image)
            source = words_tensor(words)
            try:
                raw_copy(source, target, words.shape[0])
                sync('upload_exact')
            finally:
                ttnn.deallocate(source)
            return target

        def upload(value):
            return ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(device))

        def raw_host(value):
            shape = dev.padded_shape(value.shape)
            pages = dev.page_count(value.shape)
            sink = words_tensor(torch.zeros(pages, 512, dtype=torch.int32))
            try:
                raw_copy(value, sink, pages)
                sync('raw_host')
                shards = ttnn.get_device_tensors(sink)
                images = [dev.untile_image(torch, dev.words_of(torch, ttnn.to_torch(shard)).reshape(pages, 512), shape)
                          for shard in shards]
                raw_copy(sentinel, sink, pages)
                return images
            finally:
                ttnn.deallocate(sink)

        def logical(images, shape):
            return [image[tuple(slice(0, size) for size in shape)].contiguous() for image in images]

        def release(produced):
            for output, states in produced or ():
                ttnn.deallocate(output)
                ttnn.deallocate(states)

        sentinel_operations = dev.SentinelOperations(ttnn, fill_sentinel)

        def launch(arm, groups, operations=None):
            operations = ttnn if operations is None else operations
            with watchdog.guard('launch %s' % arm):
                if arm.startswith('A'):
                    return seq.execute(device, groups, operations, output_memory=output_memory, kernels=builds[arm])
                return v5.execute(device, groups, operations, output_memory=output_memory, kernels=builds[arm])

        def coalesced_once(arm, where):
            counts = verify_trace_t1.take()
            where.setdefault('verify_t1_counts', {})[arm] = counts
            if counts != {'coalesced': 1}:
                raise AssertionError('Arm %s did not run as one coalesced launch: verify_trace_t1 counts %s'
                                     % (arm, counts))

        def read_back(arm, case, produced):
            images = []
            for user, (output, states) in enumerate(produced):
                padded, snapshots = raw_host(output), raw_host(states)
                for name, image in (('output', padded[0]), ('states', snapshots[0])):
                    pages = dev.unwritten_pages(torch, image)
                    if pages:
                        report['unwritten'].append(dict(arm=arm, case=case, user=user, tensor=name, pages=pages))
                images.append((logical(padded, output.shape), padded, snapshots))
            return images

        def moved_inputs(checks):
            moved = []
            for label, tensor, image in checks:
                if not torch.equal(dev.bits16(torch, raw_host(tensor)[0]), dev.bits16(torch, image)):
                    moved.append(dict(label))
            return moved

        class Case:
            """One host input set on the device: groups, the inputs' byte checks, and what to free."""

            def __init__(self, host):
                norm_w, users_host, poison = host
                self.owned, self.checks, self.groups = [], [], []
                self.norm_w = upload_exact(norm_w)
                self.owned.append(self.norm_w)
                self.checks.append((dict(user=None, tensor='norm_w'), self.norm_w, dev.pad_image(torch, norm_w)))
                self.names = ('qkv', 'beta', 'gate', 'initial', 'z')
                self.images = []
                for user, values in enumerate(users_host):
                    tensors = []
                    for name in self.names:
                        fill = 0.0 if poison[user] is None or name == 'initial' else poison[user]
                        image = dev.pad_image(torch, values[name], fill)
                        tensors.append(upload_exact(values[name], image))
                        self.owned.append(tensors[-1])
                        self.checks.append((dict(user=user, tensor=name), tensors[-1], image))
                        self.images.append(image)
                    self.groups.append(tuple(tensors) + (self.norm_w,))
                landed = moved_inputs(self.checks)
                if landed:
                    raise AssertionError('Inputs did not land byte for byte: %s' % landed)

            def free(self):
                for value in self.owned:
                    ttnn.deallocate(value)

        def execute_arm(arm, case, label, where):
            produced = launch(arm, case.groups, sentinel_operations)
            try:
                sync('synchronize %s' % arm)
                coalesced_once(arm, where)
                return read_back(arm, label, produced)
            finally:
                release(produced)

        def compare_users(expected, actual, users):
            per_user = []
            for user in range(users):
                parts = dict(
                    output=dev.compare(torch, expected[user][0][0], actual[user][0][0], locate_output),
                    output_padded=dev.compare(torch, expected[user][1][0], actual[user][1][0], locate_output),
                    states=dev.compare(torch, expected[user][2][0], actual[user][2][0], locate_states))
                bad_out = any(not parts[name]['exact'] for name in ('output', 'output_padded'))
                bad_states = not parts['states']['exact']
                parts['where'] = 'both' if bad_out and bad_states else 'output-only' if bad_out else \
                    'states-only' if bad_states else 'none'
                per_user.append(parts)
            return per_user

        def merged(per_user):
            return {'user%d_%s' % (user, name): value for user, parts in enumerate(per_user)
                    for name, value in parts.items() if name != 'where'}

        tallies = report['tallies'] = dict(V=dev.new_tally(), N5r=dev.new_tally(), N5x=dev.new_tally())

        def section(name, body):
            if name not in arguments.sections:
                return
            entry = report['sections'].setdefault(name, dict(error=None))
            stage('section', name=name)
            try:
                body(entry)
            except BaseException as error:  # noqa: BLE001
                entry['error'] = repr(error)
                entry['traceback'] = traceback.format_exc()
                print(entry['traceback'], flush=True)
            stage('section-done', name=name, error=entry['error'])

        verify_trace_t1.take()

        # ---------------- selftest ----------------
        def selftest(entry):
            known = torch.arange(1024 * 512, dtype=torch.int64).remainder(2 ** 31).to(torch.int32).reshape(1024, 512)
            source = words_tensor(known)
            sink = words_tensor(torch.zeros(1024, 512, dtype=torch.int32))
            try:
                raw_copy(source, sink, 1024)
                sync('selftest')
                back = dev.words_of(torch, ttnn.to_torch(ttnn.get_device_tensors(sink)[0])).reshape(1024, 512)
            finally:
                ttnn.deallocate(source)
                ttnn.deallocate(sink)
            entry['raw_round_trip_exact'] = bool(torch.equal(back, known))
            entry['env'] = report['env_read']
            if not entry['raw_round_trip_exact']:
                raise AssertionError('the raw page copy did not round-trip known bytes')

        # ---------------- p0 ----------------
        def run_pair(case, label, entry, index=0, users=None):
            """A and V on one set (alternating which goes first), V compared with A and tallied."""
            users = arguments.users if users is None else users
            order = ('A', 'V') if index % 2 == 0 else ('V', 'A')
            results = {arm: execute_arm(arm, case, label, entry) for arm in order}
            per_user = compare_users(results['A'], results['V'], users)
            parts = merged(per_user)
            exact = dev.record(tallies['V'], label, parts)
            entry.setdefault('cases', []).append(dict(case=label, exact=exact, order=list(order),
                                                      where=[p['where'] for p in per_user],
                                                      failures={n: v for n, v in parts.items() if not v['exact']}))
            moved = moved_inputs(case.checks)
            if moved:
                report['inputs_moved'].extend(dict(item, case=label) for item in moved)
            return results

        def p0(entry):
            cases = []
            for regime in arguments.regimes:
                if regime in ('R4', 'R5'):
                    continue
                for seed in (arguments.seeds if regime == 'R1' else arguments.other_seeds):
                    cases.append((regime, seed))
            for index, (regime, seed) in enumerate(cases):
                label = '%s/seed%d' % (regime, seed)
                stage('p0-case', case=label)
                case = Case(host_inputs(torch, regime, seed, arguments.users, found))
                try:
                    run_pair(case, label, entry, index)
                finally:
                    case.free()
            if 'R5' in arguments.regimes:
                entry['r5'] = []
                for rank in arguments.r5_layers:
                    label = 'R5/gdn%d' % rank
                    stage('p0-case', case=label)
                    norm_w, users_host, poison, meta = real_inputs(torch, arguments.model_dir, arguments.real_text, rank,
                                                                   found, arguments.users)
                    entry['r5'].append(meta)
                    case = Case((norm_w, users_host, poison))
                    try:
                        run_pair(case, label, entry, rank)
                    finally:
                        case.free()
            if 'R4' in arguments.regimes:
                run_r4(entry)

        def run_r4(entry):
            r4 = entry['r4'] = dict(launches=arguments.r4_launches, completed=0, exact=True, first_failure=None,
                                    inputs_checked_at=[])
            generator = torch.Generator().manual_seed(arguments.r4_seed)
            norm_w = (1 + torch.randn(1, 1, 128, generator=generator) * 0.1).bfloat16()
            norm_w_image = dev.pad_image(torch, norm_w)
            norm_w_device = upload_exact(norm_w, norm_w_image)
            carries = {arm: [(torch.randn(1, found.gdn_nv, 128, 128, generator=generator) * 0.05).bfloat16()
                             for unused in range(arguments.users)] for arm in ('A', 'V')}
            carries['V'] = list(carries['A'])
            names = ('qkv', 'beta', 'gate', 'z')
            try:
                for launch_index in range(arguments.r4_launches):
                    label = 'R4/launch%d' % launch_index
                    check = launch_index == 0 or launch_index % 16 == 15 or launch_index == arguments.r4_launches - 1
                    step = []
                    for unused in range(arguments.users):
                        step.append(dict(
                            qkv=torch.randn(1, ROWS, found.gdn_qkv, generator=generator).bfloat16(),
                            beta=torch.rand(1, ROWS, found.gdn_nv, generator=generator).bfloat16(),
                            gate=(-torch.rand(1, ROWS, found.gdn_nv, generator=generator)).bfloat16(),
                            z=torch.randn(1, ROWS, found.gdn_z, generator=generator).bfloat16()))
                    images = [{name: dev.pad_image(torch, values[name]) for name in names} for values in step]
                    shared, results, moved = [], {}, []
                    try:
                        for values, image in zip(step, images, strict=True):
                            shared.append(tuple(upload_exact(values[name], image[name]) for name in names))
                        for arm in ('A', 'V'):
                            initial_images = [dev.pad_image(torch, carries[arm][user]) for user in range(arguments.users)]
                            initials = [upload_exact(carries[arm][user], initial_images[user])
                                        for user in range(arguments.users)]
                            try:
                                groups = [(qkv, beta, gate, initials[user], z, norm_w_device)
                                          for user, (qkv, beta, gate, z) in enumerate(shared)]
                                produced = launch(arm, groups, sentinel_operations)
                                try:
                                    sync('r4 synchronize')
                                    coalesced_once(arm, r4)
                                    results[arm] = read_back(arm, label, produced)
                                finally:
                                    release(produced)
                                if check:
                                    moved.extend(moved_inputs([(dict(launch=launch_index, arm=arm, user=user,
                                                                     tensor='initial'), initials[user],
                                                                initial_images[user])
                                                               for user in range(arguments.users)]))
                            finally:
                                for value in initials:
                                    ttnn.deallocate(value)
                            carries[arm] = [results[arm][user][2][0][ROWS - 1:ROWS].clone()
                                            for user in range(arguments.users)]
                        if check:
                            checks = [(dict(launch=launch_index, user=None, tensor='norm_w'), norm_w_device,
                                       norm_w_image)]
                            checks += [(dict(launch=launch_index, user=user, tensor=name), tensors[index],
                                        images[user][name]) for user, tensors in enumerate(shared)
                                       for index, name in enumerate(names)]
                            moved.extend(moved_inputs(checks))
                            r4['inputs_checked_at'].append(launch_index)
                    finally:
                        for values in shared:
                            for value in values:
                                ttnn.deallocate(value)
                    report['inputs_moved'].extend(moved)
                    per_user = compare_users(results['A'], results['V'], arguments.users)
                    parts = merged(per_user)
                    if not dev.record(tallies['V'], label, parts) and r4['exact']:
                        r4['exact'] = False
                        r4['first_failure'] = dict(launch=launch_index, where=[p['where'] for p in per_user],
                                                   failures={n: v for n, v in parts.items() if not v['exact']})
                    r4['completed'] = launch_index + 1
                    if launch_index % 16 == 15:
                        stage('r4-progress', completed=launch_index + 1, exact=r4['exact'])
            finally:
                ttnn.deallocate(norm_w_device)
            if r4['completed'] < arguments.r4_launches:
                raise AssertionError('R4 completed %d of %d launches' % (r4['completed'], arguments.r4_launches))

        # ---------------- users ----------------
        def users_section(entry):
            entry['counts'] = []
            for count in (1, 2, 3):
                label = 'users%d/R1/seed%d' % (count, arguments.seeds[0] if arguments.seeds else 17)
                stage('users-case', case=label)
                case = Case(host_inputs(torch, 'R1', arguments.seeds[0] if arguments.seeds else 17, count, found))
                try:
                    before = len(entry.get('cases', []))
                    run_pair(case, label, entry, count, users=count)
                    entry['counts'].append(dict(users=count, exact=entry['cases'][before]['exact']))
                finally:
                    case.free()

        # ---------------- negative ----------------
        def negative(entry):
            sets = [('R1', seed) for seed in arguments.seeds] + [('RA', seed) for seed in arguments.other_seeds]
            r5 = dev.new_tally()
            for index, (regime, seed) in enumerate(sets):
                label = 'N5r/%s/seed%d' % (regime, seed)
                stage('negative-case', case=label)
                case = Case(host_inputs(torch, regime, seed, arguments.users, found))
                try:
                    results = {arm: execute_arm(arm, case, label, entry) for arm in ('A', 'N5r')}
                    dev.record(r5, label, merged(compare_users(results['A'], results['N5r'], arguments.users)))
                finally:
                    case.free()
            tallies['N5r'] = r5
            entry['N5r'] = dict(detects=dev.detects(r5), differing_bytes=r5['differing_bytes'], cases=r5['cases'],
                                exact_cases=r5['exact_cases'])
            # N5x straight after V on a DIFFERENT input set: the stale exchange pages are V's, from set X.
            x = Case(host_inputs(torch, 'R1', 1701, arguments.users, found))
            y = Case(host_inputs(torch, 'R1', 1702, arguments.users, found))
            try:
                reference = execute_arm('A', y, 'N5x/reference', entry)
                execute_arm('V', x, 'N5x/prior-V', entry)
                got = execute_arm('N5x', y, 'N5x/y', entry)
                n5x = dev.new_tally()
                dev.record(n5x, 'N5x/y', merged(compare_users(reference, got, arguments.users)))
            finally:
                x.free()
                y.free()
            tallies['N5x'] = n5x
            entry['N5x'] = dict(detects=dev.detects(n5x), differing_bytes=n5x['differing_bytes'], cases=n5x['cases'],
                                exact_cases=n5x['exact_cases'])

        # ---------------- stale ----------------
        def stale(entry):
            x = Case(host_inputs(torch, 'R1', 1801, arguments.users, found))
            y = Case(host_inputs(torch, 'R1', 1802, arguments.users, found))
            try:
                reference = {name: execute_arm('A', case, 'stale/A-' + name, entry) for name, case in (('X', x), ('Y', y))}
                outcomes = []
                for name, case in (('X', x), ('Y', y), ('X', x)):
                    got = execute_arm('V', case, 'stale/V-' + name, entry)
                    label = 'stale/V(%s)#%d' % (name, len(outcomes))
                    outcomes.append(dev.record(tallies['V'], label,
                                               merged(compare_users(reference[name], got, arguments.users))))
                entry['sequence'] = outcomes
                entry['exact'] = all(outcomes)
            finally:
                x.free()
                y.free()

        # ---------------- trace ----------------
        def trace(entry):
            x = Case(host_inputs(torch, 'R1', 1901, arguments.users, found))
            y = Case(host_inputs(torch, 'R1', 1902, arguments.users, found))
            live = Case(host_inputs(torch, 'R1', 1901, arguments.users, found))
            entry.update(exact=True, unwritten=[], replays=0, failures=[])
            traces, produced = {}, {}
            try:
                for arm in ('A', 'V'):
                    release(launch(arm, live.groups, sentinel_operations))  # compiled outside the capture
                    sync('trace warm %s' % arm)
                    verify_trace_t1.take()
                    handle = ttnn.begin_trace_capture(device, cq_id=0)
                    try:
                        try:
                            produced[arm] = launch(arm, live.groups, sentinel_operations)
                        finally:
                            ttnn.end_trace_capture(device, handle, cq_id=0)
                    except BaseException:
                        ttnn.release_trace(device, handle)
                        raise
                    traces[arm] = handle
                    coalesced_once(arm, entry)

                def stage_inputs(source):
                    for live_tensor, source_tensor in zip((t for g in live.groups for t in g[:5]),
                                                          (t for g in source.groups for t in g[:5])):
                        raw_copy(source_tensor, live_tensor, dev.page_count(live_tensor.shape))
                    sync('restage')

                for index in range(arguments.trace_replays):
                    source = x if index % 2 == 0 else y
                    stage_inputs(source)
                    got = {}
                    for arm in ('A', 'V') if index % 2 == 0 else ('V', 'A'):
                        with watchdog.guard('replay %s' % arm):
                            ttnn.execute_trace(device, traces[arm], cq_id=0, blocking=False)
                            ttnn.synchronize_device(device)
                        got[arm] = read_back(arm, 'trace%d' % index, produced[arm])
                    label = 'trace/replay%d' % index
                    tally = dev.new_tally()
                    parts = merged(compare_users(got['A'], got['V'], arguments.users))
                    exact = dev.record(tally, label, parts)
                    dev.record(tallies['V'], label, parts)
                    if not exact:
                        entry['exact'] = False
                        if len(entry['failures']) < 3:
                            entry['failures'].append(dict(replay=index, first=tally['first_failure']))
                    entry['replays'] = index + 1
                entry['unwritten'] = [item for item in report['unwritten'] if str(item.get('case', '')).startswith('trace')]
            finally:
                for handle in traces.values():
                    ttnn.release_trace(device, handle)
                for arm_produced in produced.values():
                    release(arm_produced)
                for case in (x, y, live):
                    case.free()

        # ---------------- cache ----------------
        def cache(entry):
            def entries():
                return int(device.num_program_cache_entries())

            problems, steps = [], []
            x = Case(host_inputs(torch, 'R1', 2001, arguments.users, found))
            y = Case(host_inputs(torch, 'R1', 2002, arguments.users, found))
            held = []
            try:
                reference = {name: execute_arm('A', case, 'cache/A-' + name, entry) for name, case in (('X', x), ('Y', y))}
                scratch = ttnn.empty((ROWS, found.gdn_nv, 128, 128), device=device, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                fill_sentinel(scratch)
                raw_host(scratch)
                ttnn.deallocate(scratch)

                def counted(arm, case, users, expect, name):
                    before = entries()
                    produced = launch(arm, case.groups[:users])
                    sync('cache synchronize')
                    delta = entries() - before
                    coalesced = verify_trace_t1.take()
                    steps.append(dict(step=name, arm=arm, users=users, delta=delta, expected=expect,
                                      coalesced=coalesced))
                    if delta != expect:
                        problems.append('%s: %d new program-cache entries, expected %d' % (name, delta, expect))
                    held.append(produced)
                    return read_back(arm, 'cache/' + name, produced)

                def exact_against(name, got, ref, users):
                    outcome = dev.record(tallies['V'], 'cache/' + name, merged(compare_users(ref, got, users)))
                    if not outcome:
                        problems.append('%s differs from A' % name)

                got = counted('V', x, arguments.users, 1, 'V first launch, 4 users')
                exact_against('V first', got, reference['X'], arguments.users)
                got = counted('V', y, arguments.users, 0, 'V on fresh addresses, earlier tensors held')
                exact_against('V fresh addresses', got, reference['Y'], arguments.users)
                for count in (1, 2, 3):
                    if count >= arguments.users:
                        continue  # the program at the full user count is the one already counted
                    got = counted('V', x, count, 1, 'V first launch, %d users' % count)
                    exact_against('V %d users' % count, got, reference['X'][:count], count)
                got = counted('A', x, arguments.users, 0, 'K5-A at 4 users: its own entry already exists, V must not take it over')
                for round_index in range(2):
                    got = counted('V', y, arguments.users, 0, 'V after A, round %d' % round_index)
                    exact_against('V after A %d' % round_index, got, reference['Y'], arguments.users)
                    got_a = counted('A', y, arguments.users, 0, 'A after V, round %d' % round_index)
                    outcome = dev.record(dev.new_tally(), 'cache/A-after-V', merged(
                        compare_users(reference['Y'], got_a, arguments.users)))
                    if not outcome:
                        problems.append('K5-A changed after V ran (a cache cross-hit)')
            finally:
                for produced in held:
                    release(produced)
                x.free()
                y.free()
                entry.update(steps=steps, problems=problems, ok=not problems and bool(steps))

        # ---------------- timing ----------------
        def timing(entry):
            arms = [arm for arm in arguments.timing_arms]
            orders = dev.serpentine(arms, arguments.timing_rounds)
            sets, traces, owned = [], {}, []
            entry.update(arms=arms, launches=arguments.timing_launches, rounds=arguments.timing_rounds,
                         orders=[','.join(order) for order in orders], per_launch={}, verify_t1_counts={})
            try:
                for index in range(arguments.timing_launches):
                    norm_w, users_host, unused = host_inputs(torch, 'R1', 3000 + index, arguments.users, found)
                    norm_w_device = upload(norm_w)
                    owned.append(norm_w_device)
                    groups = []
                    for values in users_host:
                        tensors = tuple(upload(values[name]) for name in ('qkv', 'beta', 'gate', 'initial', 'z'))
                        owned.extend(tensors)
                        groups.append(tensors + (norm_w_device,))
                    sets.append(groups)
                for arm in arms:
                    stage('timing-capture', arm=arm)
                    release(launch(arm, sets[0]))
                    sync('timing warm')
                    verify_trace_t1.take()
                    handle = ttnn.begin_trace_capture(device, cq_id=0)
                    try:
                        for groups in sets:
                            release(launch(arm, groups))
                    finally:
                        ttnn.end_trace_capture(device, handle, cq_id=0)
                    sync('timing capture')
                    traces[arm] = handle
                    counts = verify_trace_t1.take()
                    entry['verify_t1_counts'][arm] = counts
                    if counts != {'coalesced': arguments.timing_launches}:
                        raise AssertionError('Timing arm %s: expected %d coalesced launches, counts %s'
                                             % (arm, arguments.timing_launches, counts))
                for arm in arms:
                    for unused in range(2):
                        ttnn.execute_trace(device, traces[arm], cq_id=0, blocking=False)
                    sync('timing warm replay')
                samples = {arm: [] for arm in arms}
                stage('timing-replay')
                for order in orders:
                    for arm in order:
                        begin = time.perf_counter()
                        with watchdog.guard('timing replay %s' % arm, max(arguments.call_timeout, 600.0)):
                            ttnn.execute_trace(device, traces[arm], cq_id=0, blocking=False)
                            ttnn.synchronize_device(device)
                        samples[arm].append((time.perf_counter() - begin) * 1e6 / arguments.timing_launches)
                for arm in arms:
                    entry['per_launch'][arm] = dev.summarize_timing(samples[arm])
                entry['samples_us'] = samples
                medians = {arm: entry['per_launch'][arm]['median_us'] for arm in arms}
                entry['a_us'], entry['v_us'] = medians.get('A'), medians.get('V')
                entry.update(timing_label(entry['a_us'], entry['v_us'], medians.get('V-nosnap')))
                entry['a_nosnap_us'], entry['v_nosnap_us'] = medians.get('A-nosnap'), medians.get('V-nosnap')
                entry['v_depth1_us'] = medians.get('V-depth1')
            finally:
                for handle in traces.values():
                    ttnn.release_trace(device, handle)
                for value in owned:
                    ttnn.deallocate(value)
                sync('timing teardown')

        section('selftest', selftest)
        section('cache', cache)
        section('p0', p0)
        section('users', users_section)
        section('negative', negative)
        section('stale', stale)
        section('trace', trace)
        section('timing', timing)

        verdict, problems = decide(report)
        report.update(verdict=verdict, problems=problems)
        report['verdict_line'] = verdict_line(report)
        code = exit_code(verdict)
        summary.update(verdict=verdict, scope='reduced' if report['missing_scope'] else 'full', problems=problems,
                       p0=dict(exact_cases=tallies['V']['exact_cases'], cases=tallies['V']['cases'],
                               differing_bytes=tallies['V']['differing_bytes']),
                       negative={name: report['sections'].get('negative', {}).get(name) for name in ('N5r', 'N5x')},
                       timing={key: report['sections'].get('timing', {}).get(key)
                               for key in ('label', 'a_us', 'v_us', 'delta_us', 'v_nosnap_us', 'write_bound')},
                       v5_triple=report['v5_triple'] if verdict == 'PASS' else None,
                       a_qualified=report['a_qualified'], missing_scope=report['missing_scope'],
                       report=str(arguments.out))
        print(report['verdict_line'], flush=True)
    except BaseException as error:  # noqa: BLE001
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        report['verdict'] = 'NO-DECISION'
        summary.update(verdict='NO-DECISION', error=repr(error)[:400])
        print(report['traceback'], flush=True)
        code = 4
    finally:
        write()
        if mesh[0] is not None:
            try:
                import ttnn
                ttnn.close_mesh_device(mesh[0])
            except BaseException:  # noqa: BLE001
                pass
        summary['kind'] = KIND
        print(json.dumps(summary, default=str), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
