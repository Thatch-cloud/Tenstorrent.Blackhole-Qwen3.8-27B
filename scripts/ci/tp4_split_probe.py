"""The four-card split-invariance probe (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=split, C2_CARDS=quad).

    python3 -B /c2/scripts/ci/tp4_split_probe.py --fabric FABRIC_1D --output /probe-results/split-probe.json

THE QUESTION (docs/tp4-exact-resume.md 1.2 and 5.3): is a prefill that RESUMES at a 128-aligned position R inside a 2048-token
chunk bit-identical to the unsplit cold prefill of the same prompt? Today every prefix hit resumes on a 2048 boundary, where the
split is op-sequence identity (docs/lever-n-tp4-design-2026-10-04.md 3.4). A resume at R off 2048 runs the head segment
[R, ceil2048(R)) through the masked bucket at chunk_start=R: another matmul M, a padded bucket, and one more GDN state store at R,
"mathematically equal and not proven bit-equal". If it IS bit-equal, a decode-start checkpoint at floor128(P - 1) is an E0 (byte-exact)
design; if it is not, it is an E1 (split-twin) design that needs the owner's decision.

HOW. The image's own model code (the C2 graft plus the G1 stage: prefill_traced_chunked(start=...) and prefill_masked_bucket(chunk_start=...),
the eager TP path the fast path serves), built with the FIRST FEW real layers of the checkpoint (up to and including the first full-attention
layer, read from the checkpoint's layer_types, never assumed) from the real weights at /models (the step mounts the hub read-only), under the
profile's environment (serving_c2_contract.apply_environment of --profile from this checkout's qwen_c2_profiles.json, on top of the image's
ENV), a paged KV cache, the persistent B=1 GDN prefill scratch bound as levern_route binds it, no chunk trace. Per prompt length L
(synthetic token ids from a fixed seed; default 9,000 and 41,000) seven arms, each in its own range of KV blocks:
  A        cold: prefill_traced_chunked(tokens[:L]) from 0.
  A2       A again: determinism control (must be identical to A).
  C        split at the 2048 multiple mid = floor2048(L/2): [0, mid) then prefill_traced_chunked(start=mid). The Lever N split, proven
           identical before: the harness control (must be identical to A).
  B+128 B+1024 B+1920   split at R = mid + 128 / 1024 / 1920: [0, R) whole (its own chunks and the masked tail at floor2048(R)), the
           RoPE staged for the whole prompt, the head segment prefill_masked_bucket(tokens[R:H], chunk_start=R) with H = min(L, ceil2048(R)),
           then prefill_traced_chunked(start=H) when H < L.
  Bend     split at R = floor128(L - 300): the decode-start case (a short head segment that ends the prompt; no chunk after it).
Compared with A, bit for bit: every GDN layer's rec_state, conv_carry and conv_states (the B=1 scratch, every chip), the last-position
logits, and the K and V cache rows of [0, L) of every attention layer, each differing position classified by region (below floor2048(R),
[floor2048(R), R), [R, H), [H, L)), with the largest absolute difference (the size of the effect, for an E1 decision).

VERDICT LINE (also in the report file): SPLIT_PROBE verdict=IDENTICAL|DIFFERS|FAIL ...
  IDENTICAL  the controls hold and every 128-aligned split equals A everywhere. Exit 0.
  DIFFERS    the controls hold and at least one split differs (a valid measured answer: resumes off 2048 are E1). Exit 2.
  FAIL       a control differs (no comparison means anything), a refused split, the mesh did not open, or an error. Exit 1.
Refused before the mesh opens: an extra split (--extra-split) that is not a multiple of 128 or not inside the prompt (the flexible SDPA
q-chunk is 128 and the KV fill is block-aligned: a start off 128 is wrong SDPA, off 64 silent KV corruption).

Bounded: the default lengths and the few layers keep the arms to a few minutes; the weight load (the image's load_state_dict reads the whole
checkpoint to host, as serving does) dominates. The report is rewritten after every length, so a watchdog kill keeps what finished.

Stdlib at import; ttnn, torch and the models tree only inside run(), where a test injects fakes.
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tp4_fabric_probe  # noqa: E402
import tp4_mesh  # noqa: E402

KIND = 'split-probe-quad'
CHUNK = 2048
BLOCK = 64
QCHUNK = 128
WATCHDOG_S = 2250
DEFAULT_LENGTHS = (9000, 41000)
MIN_LENGTH = 2 * CHUNK
DEFAULT_PROFILE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
PROFILES_FILE = os.path.join(HERE, 'qwen_c2_profiles.json')
END_GAP = 300
OFFSETS = (128, 1024, 1920)
MAX_LAYERS = 8
SEED = 20261011
CONTROLS = ('A2', 'C')
ENV_OF_INTEREST = ('QWEN35_GDN_STATE_BF16', 'QWEN_SDPA_BF8', 'QWEN_FAST_GDN_PREFILL_CONV', 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT',
                   'QWEN_FAST_SDPA_PF', 'QWEN_FAST_SDPA_PF_ONEQ', 'QWEN_FAST_GRID13', 'TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE')


def floor_to(value, multiple):
    return (int(value) // multiple) * multiple


def ceil_to(value, multiple):
    return -(-int(value) // multiple) * multiple


def check_split(length, split):
    """None when `split` is a resume point this probe may run for a prompt of `length`, else why not."""
    if type(split) is not int or type(length) is not int:
        return 'split and length must be integers'
    if not 0 < split < length:
        return 'R=%d is not inside the prompt (0, %d)' % (split, length)
    if split % QCHUNK:
        return ('R=%d is not a multiple of %d: the flexible chunked SDPA runs q_chunk=%d (chunk_start %% q_chunk == 0) and the KV fill '
                'writes whole %d-token blocks from slot 0, so a start off %d is wrong SDPA and off %d silent KV corruption'
                % (split, QCHUNK, QCHUNK, BLOCK, QCHUNK, BLOCK))
    return None


def plan_arms(length, extra=()):
    """[(name, R or None)] for a prompt of `length`: A, A2 (None: cold), C (2048-aligned control), the 128-aligned splits."""
    if type(length) is not int or length < MIN_LENGTH:
        raise ValueError('a probe length must be an integer >= %d, got %r' % (MIN_LENGTH, length))
    mid = floor_to(length // 2, CHUNK)
    arms = [('A', None), ('A2', None), ('C', mid)]
    for offset in OFFSETS:
        arms.append(('B+%d' % offset, mid + offset))
    end = floor_to(length - END_GAP, QCHUNK)
    if end % CHUNK == 0:
        end -= QCHUNK
    arms.append(('Bend', end))
    for split in extra:
        if split < length:
            arms.append(('Bx%d' % split, split))
    for name, split in arms:
        if split is not None:
            why = check_split(length, split)
            if why:
                raise ValueError('%s: %s' % (name, why))
    return arms


def regions(split, length):
    """The four position ranges a split's differences are classified into."""
    if split is None:
        return []
    low = floor_to(split, CHUNK)
    head_end = split if split % CHUNK == 0 else min(length, ceil_to(split, CHUNK))
    return [('below_floor2048_R', 0, low), ('floor2048_R_to_R', low, split), ('R_to_H', split, head_end), ('H_to_L', head_end, length)]


def same_bits(torch, left, right):
    """Bit equality of two host tensors (shape, dtype and every byte)."""
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    a = left.contiguous().reshape(-1).view(torch.uint8)
    b = right.contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(a, b))


def max_abs(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return None
    try:
        diff = (left.float() - right.float()).abs()
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        return float(diff.max()) if diff.numel() else 0.0
    except Exception:  # noqa: BLE001
        return None


def compare_state(torch, ref, got):
    """GDN scratch: {'equal', 'tensors', 'differing', 'max_abs'} over every layer's rec_state, conv_carry and conv_states."""
    tensors = differing = 0
    worst = 0.0
    names = []
    for layer, (a_layer, b_layer) in enumerate(zip(ref, got)):
        for key in sorted(set(a_layer) | set(b_layer)):
            a_list, b_list = a_layer.get(key) or [], b_layer.get(key) or []
            if len(a_list) != len(b_list):
                differing += 1
                names.append('layer%d.%s(count)' % (layer, key))
                continue
            for index, (a, b) in enumerate(zip(a_list, b_list)):
                tensors += 1
                if not same_bits(torch, a, b):
                    differing += 1
                    names.append('layer%d.%s[%d]' % (layer, key, index))
                    value = max_abs(torch, a, b)
                    worst = max(worst, value or 0.0)
    if len(ref) != len(got):
        differing += 1
        names.append('layer count %d != %d' % (len(ref), len(got)))
    return dict(equal=differing == 0, tensors=tensors, differing=differing, max_abs=worst, which=names[:24])


def compare_logits(torch, ref, got):
    equal = same_bits(torch, ref, got)
    out = dict(equal=equal, max_abs=0.0 if equal else max_abs(torch, ref, got))
    try:
        out['argmax_equal'] = bool(int(ref.float().reshape(-1).argmax()) == int(got.float().reshape(-1).argmax()))
    except Exception:  # noqa: BLE001
        out['argmax_equal'] = None
    return out


def kv_rows(torch, cache, base, length):
    """[positions, heads, head_dim] rows [0, length) of an arm whose logical block j is physical block base + j.
    `cache` is the host cache [blocks, heads (every chip), 64, head_dim]."""
    blocks = ceil_to(length, BLOCK) // BLOCK
    part = cache[base:base + blocks]
    rows = part.permute(0, 2, 1, 3).reshape(blocks * BLOCK, part.shape[1], part.shape[3])
    return rows[:length]


def compare_kv(torch, ref_caches, got_caches, ref_base, got_base, length, split):
    """Per attention layer, K and V: the positions in [0, length) whose rows differ, counted per region of the split."""
    out = dict(equal=True, layers=[])
    for layer, ((ref_k, ref_v), (got_k, got_v)) in enumerate(zip(ref_caches, got_caches)):
        entry = dict(layer=layer)
        for name, ref, got in (('k', ref_k, got_k), ('v', ref_v, got_v)):
            a = kv_rows(torch, ref, ref_base, length)
            b = kv_rows(torch, got, got_base, length)
            if tuple(a.shape) != tuple(b.shape):
                entry[name] = dict(equal=False, error='shape %s != %s' % (tuple(a.shape), tuple(b.shape)))
                out['equal'] = False
                continue
            per_position = (a.contiguous().view(torch.uint8).reshape(length, -1) != b.contiguous().view(torch.uint8).reshape(length, -1)).any(dim=1)
            count = int(per_position.sum())
            detail = dict(equal=count == 0, differing_positions=count)
            if count:
                out['equal'] = False
                positions = per_position.nonzero().reshape(-1)
                detail['first'] = int(positions[0])
                detail['last'] = int(positions[-1])
                detail['regions'] = {label: int(per_position[low:high].sum()) for label, low, high in regions(split, length) if high > low}
                detail['max_abs'] = max_abs(torch, a, b)
            entry[name] = detail
        out['layers'].append(entry)
    return out


def verdict(lengths):
    """(verdict, reasons) over the per-length results: FAIL unless every control holds; then IDENTICAL or DIFFERS."""
    reasons = []
    splits_differ = []
    if not lengths:
        return 'FAIL', ['no length was measured']
    for item in lengths:
        if item.get('error'):
            reasons.append('L=%s: %s' % (item.get('length'), item['error']))
            continue
        for arm in item.get('arms', []):
            if arm['name'] == 'A':
                continue
            if arm.get('error'):
                reasons.append('L=%d %s: %s' % (item['length'], arm['name'], arm['error']))
                continue
            if arm['name'] in CONTROLS:
                if not arm['identical']:
                    reasons.append('L=%d control %s differs from A: the harness or the device is not deterministic' % (item['length'], arm['name']))
            elif not arm['identical']:
                splits_differ.append('L=%d %s (R=%d)' % (item['length'], arm['name'], arm['split']))
    if reasons:
        return 'FAIL', reasons
    if splits_differ:
        return 'DIFFERS', splits_differ
    return 'IDENTICAL', []


def apply_profile(profile_name, environ, path=PROFILES_FILE):
    """The profile's environment on top of the image's, as serving applies it; the step's mesh descriptor and device are kept."""
    import serving_c2_contract

    profile = serving_c2_contract.load_profile(path, profile_name)
    keep = dict((name, environ.get(name)) for name in ('TT_MESH_GRAPH_DESC_PATH', 'MESH_DEVICE'))
    serving_c2_contract.apply_environment(profile, environ)
    for name, value in keep.items():
        if value is None:
            environ.pop(name, None)
        else:
            environ[name] = value
    snapshot = serving_c2_contract.resolve_snapshot(profile)
    environ['HF_MODEL'] = snapshot
    environ.setdefault('HF_HUB_OFFLINE', '1')
    return profile, snapshot


def pick_layers(types):
    """Checkpoint layer indices: every layer up to and including the first full-attention one (at most MAX_LAYERS), so the subset has
    GDN layers and one attention layer whatever the checkpoint's pattern."""
    types = list(types)
    for index, kind in enumerate(types[:MAX_LAYERS]):
        if kind == 'full_attention':
            picked = list(range(index + 1))
            if not any(types[i] == 'linear_attention' for i in picked):
                raise ValueError('no linear-attention (GDN) layer before the first full-attention layer: %r' % types[:MAX_LAYERS])
            return picked
    raise ValueError('no full-attention layer in the first %d layer types: %r' % (MAX_LAYERS, types[:MAX_LAYERS]))


def build_model(ttnn, mesh, options, environ, log):
    """The image's model with the first few real layers: Qwen36ModelArgs + load_state_dict + Qwen36Model, as from_pretrained builds it
    (layer_indices keeps each layer's real checkpoint index and type), with no tensor cache (the probe's /tmp is small and must not
    write the persistent cache). Returns (model, info)."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=1, max_seq_len=options.max_seq_len)
    indices = pick_layers(args.attention_type_list)
    args.layer_indices = indices
    args.n_layers = len(indices)
    began = time.time()
    state_dict = args.load_state_dict()
    loaded_s = time.time() - began
    # The state dict above is the real one. dummy_weights only stops the tt_transformers modules the model reuses (the Embedding) from
    # naming a cache file under a None path; the Qwen modules take tensor_cache_path=None as "no cache" themselves (graft layer.py, mlp.py).
    args.dummy_weights = True
    cache = 'none'
    try:
        model = Qwen36Model(mesh, args, state_dict, tensor_cache_path=None)
    except TypeError as error:
        # A module that still joins a None path: cache under the step's 8 GB compile-cache tmpfs instead (never the persistent cache).
        from pathlib import Path

        root = Path(environ.get('TT_METAL_CACHE') or '/root/.cache/tt-metal-cache') / 'split-probe-weights'
        root.mkdir(parents=True, exist_ok=True)
        log('SPLIT_PROBE model without a tensor cache failed (%s); retrying with the cache at %s' % (error, root))
        gc.collect()
        model = Qwen36Model(mesh, args, state_dict, tensor_cache_path=root)
        cache = str(root)
    del state_dict
    gc.collect()
    info = dict(layer_indices=indices, layer_types=[args.attention_type_list[i] for i in indices], load_state_dict_s=round(loaded_s, 1),
                tensor_cache=cache,
                build_s=round(time.time() - began, 1), n_local_kv_heads=int(args.n_local_kv_heads), head_dim=int(args.head_dim),
                vocab_size=int(args.vocab_size))
    log('SPLIT_PROBE model layers=%s types=%s load_s=%.1f build_s=%.1f' % (indices, info['layer_types'], loaded_s, info['build_s']))
    return model, info


def host(ttnn, tensor, composer):
    return ttnn.to_torch(tensor, mesh_composer=composer)


def read_state(ttnn, model, composer):
    """Every GDN layer's scratch state (the scratch must be bound), as host tensors."""
    out = []
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        entry = dict(rec_state=[host(ttnn, dn.rec_state, composer)])
        if getattr(dn, 'conv_carry', None) is not None:
            entry['conv_carry'] = [host(ttnn, dn.conv_carry, composer)]
        entry['conv_states'] = [host(ttnn, state, composer) for state in (getattr(dn, 'conv_states', None) or [])]
        out.append(entry)
    return out


def read_kv(ttnn, model, heads):
    return [[host(ttnn, k, heads), host(ttnn, v, heads)] for k, v in model._paged_kv_caches]


def logits_of(ttnn, mesh, logits, composer):
    value = host(ttnn, logits, composer)
    try:
        ttnn.deallocate(logits)
    except Exception:  # noqa: BLE001
        pass
    return value


def run_arm(ttnn, torch, model, mesh, tokens, length, split, page_table, composer):
    """One arm into the bound scratch and its own KV block range; returns the host logits."""
    if split is None:
        logits = model.prefill_traced_chunked(tokens[:, :length], page_table, actual_len=length)
        ttnn.synchronize_device(mesh)
        return logits_of(ttnn, mesh, logits, composer)
    first = model.prefill_traced_chunked(tokens[:, :split], page_table, actual_len=split)
    ttnn.synchronize_device(mesh)
    logits_of(ttnn, mesh, first, composer)
    if split % CHUNK == 0:
        logits = model.prefill_traced_chunked(tokens[:, :length], page_table, actual_len=length, start=split)
        ttnn.synchronize_device(mesh)
        return logits_of(ttnn, mesh, logits, composer)
    head_end = min(length, ceil_to(split, CHUNK))
    # The masked bucket at chunk_start > 0 keeps the staged RoPE table: stage the whole prompt's, as the cold run's entry does.
    model._build_request_rope(tokens[:, :length], None)
    logits = model.prefill_masked_bucket(tokens[:, split:head_end], page_table, actual_len=head_end - split, chunk_start=split)
    ttnn.synchronize_device(mesh)
    if head_end < length:
        logits_of(ttnn, mesh, logits, composer)
        logits = model.prefill_traced_chunked(tokens[:, :length], page_table, actual_len=length, start=head_end)
        ttnn.synchronize_device(mesh)
    return logits_of(ttnn, mesh, logits, composer)


def measure_length(ttnn, torch, model, mesh, length, width, seed, extra, log):
    """The seven (or more) arms of one length, each in its own KV block range, compared with A."""
    arms = plan_arms(length, extra)
    generator = torch.Generator().manual_seed(seed + length)
    vocab = int(getattr(model, 'vocab_size', None) or model.args.vocab_size)
    tokens = torch.randint(0, vocab, (1, length), generator=generator, dtype=torch.int64).to(torch.int32)
    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
    heads = ttnn.ConcatMeshToTensor(mesh, dim=1)
    results, states, logits = [], {}, {}
    previous = model._bind_gdn_prefill_scratch()
    try:
        for index, (name, split) in enumerate(arms):
            base = 1 + index * width
            page_table = torch.arange(base, base + width, dtype=torch.int32).reshape(1, width)
            began = time.time()
            try:
                logits[name] = run_arm(ttnn, torch, model, mesh, tokens, length, split, page_table, composer)
                states[name] = read_state(ttnn, model, composer)
                results.append(dict(name=name, split=split, base=base, seconds=round(time.time() - began, 2)))
            except Exception as error:  # noqa: BLE001
                results.append(dict(name=name, split=split, base=base, error='%s: %s' % (type(error).__name__, str(error)[:400]),
                                    traceback=traceback.format_exc()[-2000:]))
                log('SPLIT_PROBE arm L=%d %s failed: %s' % (length, name, results[-1]['error']))
                if name == 'A':
                    break
    finally:
        model._unbind_gdn_prefill_scratch(previous)
    caches = read_kv(ttnn, model, heads)
    reference = next((item for item in results if item['name'] == 'A' and not item.get('error')), None)
    if reference is None:
        return dict(length=length, error='the cold arm A did not run', arms=results)
    for item in results:
        if item['name'] == 'A' or item.get('error'):
            continue
        name = item['name']
        state = compare_state(torch, states['A'], states[name])
        lg = compare_logits(torch, logits['A'], logits[name])
        kv = compare_kv(torch, caches, caches, reference['base'], item['base'], length, item['split'])
        item.update(gdn=state, logits=lg, kv=kv, identical=bool(state['equal'] and lg['equal'] and kv['equal']))
        log('SPLIT_PROBE arm L=%d %s R=%s identical=%s gdn_differing=%d/%d gdn_max_abs=%s logits_equal=%s argmax_equal=%s kv_equal=%s' % (
            length, name, item['split'], item['identical'], state['differing'], state['tensors'], state['max_abs'], lg['equal'],
            lg.get('argmax_equal'), kv['equal']))
    return dict(length=length, arms=results, regions={item['name']: regions(item['split'], length) for item in results if item['split']})


def default_torch():
    import torch

    return torch


def run(options, ttnn=None, torch=None, model_factory=None, environ=None, log=print, checkpoint=None):
    """The measurement; returns the report dict. `checkpoint(report)` is called after every length (the incremental report)."""
    environ = os.environ if environ is None else environ
    report = dict(kind=KIND, fabric=options.fabric, profile=options.profile, lengths=list(options.lengths), extra_splits=list(options.extra_split),
                  opened=False, descriptor=environ.get('TT_MESH_GRAPH_DESC_PATH'), serving_contract=environ.get('QWEN_C2_SERVING'))
    for length in options.lengths:
        try:
            plan_arms(length, options.extra_split)
        except ValueError as error:
            report.update(verdict='FAIL', error='refused: %s' % error)
            log('SPLIT_PROBE verdict=FAIL error=%s' % report['error'])
            return report
    if ttnn is None:
        descriptor = report['descriptor']
        problems = None
        if descriptor and os.path.isfile(descriptor):
            with open(descriptor, encoding='utf-8') as handle:
                problems = tp4_mesh.descriptor_problems(handle.read())
            report['descriptor_problems'] = problems
        refusal = tp4_fabric_probe.descriptor_refusal(descriptor, problems)
        if refusal:
            report.update(verdict='FAIL', error='refused to open: ' + refusal)
            log('SPLIT_PROBE verdict=FAIL error=%s' % report['error'])
            return report
        try:
            _, snapshot = apply_profile(options.profile, environ)
            report['snapshot'] = os.path.basename(snapshot.rstrip('/'))
        except Exception as error:  # noqa: BLE001
            report.update(verdict='FAIL', error='profile: %s: %s' % (type(error).__name__, error))
            log('SPLIT_PROBE verdict=FAIL error=%s' % report['error'])
            return report
        import ttnn
    torch = default_torch() if torch is None else torch
    model_factory = build_model if model_factory is None else model_factory
    report['env'] = dict((name, environ.get(name)) for name in ENV_OF_INTEREST)
    mesh = None
    try:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
        report['opened'] = True
        report['chips'] = int(mesh.get_num_devices())
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [int(grid.x), int(grid.y)]
        model, info = model_factory(ttnn, mesh, options, environ, log)
        report['model'] = info
        width = ceil_to(ceil_to(max(options.lengths) + CHUNK, BLOCK) // BLOCK, 32)
        arms = max(len(plan_arms(length, options.extra_split)) for length in options.lengths)
        blocks = 1 + arms * width
        shape = (blocks, int(info['n_local_kv_heads']), BLOCK, int(info['head_dim']))
        model.allocate_kv_caches(shape, ttnn.bfloat16, batch_size=1)
        report['kv_cache_shape'] = list(shape)
        dtype = None
        for layer in model.layers:
            if not layer.is_full_attention:
                dtype = str(getattr(layer.attention.rec_state, 'dtype', None))
                break
        report['rec_state_dtype'] = dtype
        report['results'] = []
        for length in options.lengths:
            began = time.time()
            try:
                item = measure_length(ttnn, torch, model, mesh, length, width, options.seed, options.extra_split, log)
            except Exception as error:  # noqa: BLE001
                item = dict(length=length, error='%s: %s' % (type(error).__name__, str(error)[:500]), traceback=traceback.format_exc()[-3000:])
            item['seconds'] = round(time.time() - began, 1)
            report['results'].append(item)
            if checkpoint is not None:
                checkpoint(report)
        report['verdict'], report['reasons'] = verdict(report['results'])
    except Exception as error:  # noqa: BLE001
        report.update(verdict='FAIL', error='%s: %s' % (type(error).__name__, str(error)[:500]), traceback=traceback.format_exc()[-3000:])
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except Exception as error:  # noqa: BLE001
                report['closed'] = '%s: %s' % (type(error).__name__, error)
    splits = sum(1 for item in report.get('results', []) for arm in item.get('arms', []) if arm.get('split') and arm['name'] not in CONTROLS)
    differing = sum(1 for item in report.get('results', []) for arm in item.get('arms', [])
                    if arm.get('split') and arm['name'] not in CONTROLS and arm.get('identical') is False)
    log('SPLIT_PROBE verdict=%s lengths=%s splits=%d differing=%d grid=%s layers=%s rec_state_dtype=%s%s' % (
        report.get('verdict'), ','.join(str(length) for length in options.lengths), splits, differing, report.get('grid'),
        (report.get('model') or {}).get('layer_indices'), report.get('rec_state_dtype'),
        (' error=%s' % report['error']) if report.get('error') else ''))
    for reason in report.get('reasons') or []:
        log('SPLIT_PROBE reason: %s' % reason)
    return report


def lengths_of(text):
    values = [int(part) for part in str(text).replace(',', ' ').split()]
    if not values:
        raise argparse.ArgumentTypeError('at least one length')
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=tp4_mesh.FABRIC_CONFIGS, default=tp4_mesh.FABRIC_CONFIG)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', default=DEFAULT_PROFILE)
    parser.add_argument('--lengths', type=lengths_of, default=list(DEFAULT_LENGTHS))
    parser.add_argument('--extra-split', type=int, action='append', default=[])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--max-seq-len', type=int, default=65536)
    return parser


def write_report(path, report):
    with open(path + '.tmp', 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    os.replace(path + '.tmp', path)


def main(argv=None, ttnn=None, torch=None, model_factory=None, environ=None, log=print):
    options = build_parser().parse_args(argv)
    if any(length >= options.max_seq_len for length in options.lengths):
        print('refusing: every length must be below --max-seq-len %d' % options.max_seq_len, file=sys.stderr)
        return 1
    timer = threading.Timer(WATCHDOG_S, lambda: (print('SPLIT_PROBE verdict=FAIL error=watchdog after %d s' % WATCHDOG_S, flush=True),
                                                 os._exit(1)))
    timer.daemon = True
    timer.start()
    try:
        report = run(options, ttnn=ttnn, torch=torch, model_factory=model_factory, environ=environ, log=log,
                     checkpoint=lambda partial: write_report(options.output, partial))
        write_report(options.output, report)
        log(json.dumps(dict((key, report.get(key)) for key in ('kind', 'verdict', 'reasons', 'grid', 'model', 'rec_state_dtype', 'error')),
                       sort_keys=True, default=str))
    finally:
        timer.cancel()
    return {'IDENTICAL': 0, 'DIFFERS': 2}.get(report.get('verdict'), 1)


if __name__ == '__main__':
    sys.exit(main())
