"""WP-F3: the ttml feasibility probe, the ordered stages that decide whether the drafter fine-tune can train on our own cards through
tt-train's Python package (ttml) or needs a hand-written ttnn backward. The stage logic, thresholds, references and the report are
here and are tested on CPU against a fake ttml; the REAL adapter (`Ops` backed by ttml / ttnn on the four-card mesh) is written at the
first card window against the pinned tree and is not part of this change.

THE STAGES (write counts and numbers only; no host, no serial, no path, no text):
  P0  import ttml and see the devices
  P1  one card, a tiny Qwen3-shaped forward and backward: the loss must fall
  P2  DDP4 all-reduce on our 1x4 descriptor: GB/s per pair against the measured 84-90 GB/s
  P3  op probes, each compared with the CPU reference of the F2 training code (dflash2_torch / ft_*):
        sdpa_square_mask at S 6.5k and 8.6k, per-row RoPE, the grouped dynamic convolution, chunked cross-entropy at the
        248,320 vocabulary, ttnn.topk (k = 16); plus the memory of the composite rectangular attention path (informational: if it does
        not fit, the answer is the square kernel with an arbitrary mask, which the first probe covers)
  P4  the #41657 check: the FULL view and the NATIVE / HALF view of a parameter after fused bf16 AdamW steps must agree
  P5  TFLOPS per card of a DFlash2-shaped layer at B = 16 x 512 anchors
  P6  DDP4 bf16 memory per card against the estimate of about 22 GB

DECISION (design train-on-tt): GO for the ttml route when P0-P4 pass and P5 reaches 15 TFLOPS per card; otherwise FALLBACK (a
hand-written ttnn backward, +10-15 eng-days, or stop). P6 is reported, not gating. A gating stage that did not run leaves the decision
NOT_ESTABLISHED.

Also here: `validate_mesh` (the four-card descriptor and its link counts, through tp4_mesh) and `check_backports` (the manifest of the
upstream fixes the image must carry: patches/tt-train/backports.json).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tau_lab_report as rep  # noqa: E402

STAGES = ('P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6')
GATING = ('P0', 'P1', 'P2', 'P3', 'P4')
OPS = ('sdpa_square_mask_6500', 'sdpa_square_mask_8600', 'per_row_rope', 'grouped_conv', 'chunked_ce_248320', 'topk16')
INFORMATIONAL = ('composite_rect_memory',)
THRESHOLDS = dict(p1_loss_ratio=0.9, p2_gbps_min=60.0, p3_relative_error=2e-2, p4_relative_error=1e-2, p5_tflops_min=15.0,
                  p6_gib_max=26.0, rect_gib_max=3.0)
WORDS = frozenset(('PASS', 'FAIL', 'NOT_RUN', 'GO', 'FALLBACK', 'NOT_ESTABLISHED', 'yes', 'no'))
SDPA_SIZES = dict(sdpa_square_mask_6500=6500, sdpa_square_mask_8600=8600)


class ProbeError(ValueError):
    pass


# -- the references (the F2 CPU code) ---------------------------------------------------------------------------------------------

def reference_inputs(name, seed=0, sizes=None):
    """Deterministic inputs for an op probe, and the CPU reference output, as (inputs dict, expected tensor)."""
    import torch
    import dflash2_torch as d2
    generator = torch.Generator().manual_seed(seed)
    sizes = sizes or {}
    if name in SDPA_SIZES:
        length = sizes.get(name, SDPA_SIZES[name])
        heads, dim = 2, 64
        q, k, v = (torch.randn(1, heads, length, dim, generator=generator) for _ in range(3))
        mask = torch.rand(1, 1, length, length, generator=generator) > 0.5
        mask |= torch.eye(length, dtype=torch.bool)[None, None]
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return dict(q=q, k=k, v=v, mask=mask), expected
    if name == 'per_row_rope':
        length, dim = sizes.get(name, 256), 64
        x = torch.randn(1, 4, length, dim, generator=generator)
        positions = torch.sort(torch.randint(0, 100000, (1, length), generator=generator), dim=1).values   # arbitrary, not contiguous
        cos, sin = d2.rotary(positions, dim, 1e7, torch.float32)
        return dict(x=x, positions=positions, theta=1e7), d2.apply_rope(x, cos, sin)
    if name == 'grouped_conv':
        hidden_size, block, blocks = 64, 8, sizes.get(name, 6)
        hidden = torch.randn(1, blocks * block, hidden_size, generator=generator)
        conv = d2.GroupedDynamicCausalConv(hidden_size, 2, 16)
        with torch.no_grad():
            conv.base_kernel.copy_(torch.randn(conv.base_kernel.shape, generator=generator))
            conv.kernel_projection.weight.copy_(0.3 * torch.randn(conv.kernel_projection.weight.shape, generator=generator))
            pre, _ = conv.prepare(hidden, block)
        return dict(hidden=hidden, base_kernel=conv.base_kernel.detach().clone(), projection=conv.kernel_projection.weight.detach().clone(),
                    block=block, group=16), pre
    if name == 'chunked_ce_248320':
        rows, dim, vocab = sizes.get(name, 64), 32, 248320
        hidden = torch.randn(rows, dim, generator=generator)
        weight = torch.randn(vocab, dim, generator=generator)
        targets = torch.randint(0, vocab, (rows,), generator=generator)
        losses = torch.nn.functional.cross_entropy(hidden @ weight.T, targets, reduction='none')
        return dict(hidden=hidden, weight=weight, targets=targets), losses
    if name == 'topk16':
        rows, vocab = sizes.get(name, 32), 248320
        logits = torch.randn(rows, vocab, generator=generator)
        return dict(logits=logits, k=16), torch.topk(logits, 16, dim=-1).values
    raise ProbeError('unknown op probe')


def relative_error(got, expected):
    import torch
    got, expected = got.float(), expected.float()
    if got.shape != expected.shape or not torch.isfinite(got).all():
        return float('inf')
    return float((got - expected).abs().max() / (expected.abs().max() + 1e-12))


# -- the stages -------------------------------------------------------------------------------------------------------------------

def stage_p0(ops, limits):
    imported, devices = ops.import_ok(), ops.devices()
    return dict(status='PASS' if imported and devices >= 1 else 'FAIL', import_ok=bool(imported), devices=int(devices))


def stage_p1(ops, limits):
    losses = ops.tiny_train(steps=20)
    if not losses or any(loss != loss for loss in losses):
        return dict(status='FAIL', steps=len(losses or []), loss_ratio=None)
    ratio = losses[-1] / losses[0] if losses[0] else None
    return dict(status='PASS' if ratio is not None and ratio <= limits['p1_loss_ratio'] else 'FAIL', steps=len(losses),
                loss_ratio=round(ratio, 4) if ratio is not None else None)


def stage_p2(ops, limits):
    gbps = ops.allreduce_gbps()
    return dict(status='PASS' if gbps >= limits['p2_gbps_min'] else 'FAIL', gbps_per_pair=round(gbps, 1))


def stage_p3(ops, limits, sizes=None):
    results, ok = {}, True
    for name in OPS:
        try:
            inputs, expected = reference_inputs(name, sizes=sizes)
            error = relative_error(ops.run_op(name, inputs), expected)
            passed = error <= limits['p3_relative_error']
            results[name] = dict(relative_error=round(error, 6) if error != float('inf') else None, passed=passed)
        except Exception:                                  # a probe that raises is a failed probe; the type stays out of the report
            results[name] = dict(relative_error=None, passed=False, raised=True)
            passed = False
        ok = ok and passed
    rect = ops.rect_attention_gib()
    results['composite_rect_memory'] = dict(gib=round(rect, 2) if rect is not None else None,
                                            fits=bool(rect is not None and rect <= limits['rect_gib_max']))
    return dict(status='PASS' if ok else 'FAIL', ops=results)


def stage_p4(ops, limits):
    error = ops.adam_view_error()
    return dict(status='PASS' if error is not None and error <= limits['p4_relative_error'] else 'FAIL',
                relative_error=round(error, 6) if error is not None else None)


def stage_p5(ops, limits):
    tflops = ops.layer_tflops()
    return dict(status='PASS' if tflops >= limits['p5_tflops_min'] else 'FAIL', tflops_per_card=round(tflops, 1))


def stage_p6(ops, limits):
    gib = ops.ddp_memory_gib()
    return dict(status='PASS' if gib <= limits['p6_gib_max'] else 'FAIL', gib_per_card=round(gib, 1))


STAGE_FUNCTIONS = dict(P0=stage_p0, P1=stage_p1, P2=stage_p2, P3=stage_p3, P4=stage_p4, P5=stage_p5, P6=stage_p6)


def decide(stages):
    """GO / FALLBACK / NOT_ESTABLISHED from the stage results {name: {status: ...}}. Nothing is concluded when ttml cannot even be
    imported (P0: an environment problem, not a verdict on the route); a gating stage that ran and failed is a FALLBACK even when later
    ones did not run; a gating stage that did not run, with none failed, leaves it NOT_ESTABLISHED."""
    gating = GATING + ('P5',)
    if stages.get('P0', {}).get('status') != 'PASS':
        return 'NOT_ESTABLISHED'
    if any(stages.get(name, {}).get('status') == 'FAIL' for name in gating):
        return 'FALLBACK'
    if any(stages.get(name, {}).get('status') != 'PASS' for name in gating):
        return 'NOT_ESTABLISHED'
    return 'GO'


def run(ops, limits=None, stop_after_p1_failure=True, sizes=None):
    """The stages in order. P0 and P1 gate the rest (a missing import or a loss that does not fall makes every later number
    meaningless): a failure there leaves the later stages NOT_RUN. P2-P6 always run, so one window gathers everything it can."""
    limits = dict(THRESHOLDS, **(limits or {}))
    stages = {}
    stop = False
    for name in STAGES:
        if stop:
            stages[name] = dict(status='NOT_RUN')
            continue
        try:
            result = STAGE_FUNCTIONS[name](ops, limits, sizes) if name == 'P3' else STAGE_FUNCTIONS[name](ops, limits)
        except Exception:
            result = dict(status='FAIL', raised=True)
        stages[name] = result
        if name in ('P0', 'P1') and result['status'] != 'PASS' and stop_after_p1_failure:
            stop = True
    return dict(stages=stages, decision=decide(stages), limits=dict((k, v) for k, v in limits.items()))


def public_report(report):
    """The report with every non-numeric leaf checked: stage statuses, decisions and yes / no only."""
    rep.assert_public(report, words=WORDS)
    return report


# -- the mesh and the backports ---------------------------------------------------------------------------------------------------

def validate_mesh(text, devices=4, links=2):
    """Problems (a list of plain strings, empty when fine) with a mesh graph descriptor for the four-card training mesh: it must be the
    descriptor tp4_mesh describes (2 x 2 devices, `links` channels per edge) and must carry the link count of our measured fabric."""
    import tp4_mesh
    problems = list(tp4_mesh.descriptor_problems(text, links))
    try:
        dims = tp4_mesh.parse_descriptor(text)['dims']
    except ValueError:
        return problems                      # descriptor_problems already said it is unreadable
    count = 1
    for value in dims:
        count *= value
    if count != devices:
        problems.append('the descriptor opens %d devices, %d expected' % (count, devices))
    return problems


REQUIRED_FIELDS = ('pr', 'purpose', 'status')
STATUSES = ('not_fetched', 'ready', 'merged_upstream')


def check_backports(manifest, directory):
    """Problems with patches/tt-train/backports.json: every entry named, with a known status, and a patch file present for each that
    claims to be ready. An empty list means the image build may apply them in order."""
    problems = []
    entries = manifest.get('backports') if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        return ['the manifest lists no backports']
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or any(field not in entry for field in REQUIRED_FIELDS):
            problems.append('an entry lacks %s' % ', '.join(REQUIRED_FIELDS))
            continue
        if entry['pr'] in seen:
            problems.append('PR %s is listed twice' % entry['pr'])
        seen.add(entry['pr'])
        if entry['status'] not in STATUSES:
            problems.append('PR %s has an unknown status' % entry['pr'])
        elif entry['status'] == 'ready' and not os.path.isfile(os.path.join(directory, '%s.patch' % entry['pr'])):
            problems.append('PR %s is ready but its patch file is missing' % entry['pr'])
    return problems


def main(argv=None, say=print, make_ops=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', required=True, help='where the counts-only JSON report is written')
    parser.add_argument('--keep-going', action='store_true', help='run P2-P6 even when P0 or P1 failed')
    options = parser.parse_args(argv)
    try:
        ops = make_ops() if make_ops else _real_ops()
        report = public_report(run(ops, stop_after_p1_failure=not options.keep_going))
    except Exception as error:               # the type only
        say('refused: %s' % type(error).__name__)
        return 2
    with open(options.out, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
        handle.write('\n')
    for name in STAGES:
        say('%s %s' % (name, report['stages'][name]['status']))
    say('decision %s' % report['decision'])
    return 0 if report['decision'] == 'GO' else 1


def _real_ops():  # pragma: no cover - the adapter is written at the first card window against the pinned tree
    raise NotImplementedError('the ttml adapter is not part of this change')


if __name__ == '__main__':
    sys.exit(main())
