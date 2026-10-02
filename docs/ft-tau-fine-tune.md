# Drafter fine-tune on our own cards: what is built, what is gated

Status: the first packages that need no card and no new data intake are built and tested on CPU. Nothing here has touched a card,
the rig, production or any model weight. Training happens on our own Tenstorrent cards (the owner's decision; no external GPU).

## The gate comes first (K2), and it fails on measured data

The design's K2 test projects the central fine-tune (x1.15-1.33) and lookup (x1.03-1.05) onto the tau lab's measured worst-seat
statistics. On W-T1a (thinking on): per-turn p10 3.47 -> 4.11-4.85; worst-of-8 p10 2.92 -> 3.46-4.08; the 80k+ bucket 3.83 -> 4.54-5.35.
Every projection is below the 5.95 / 7.6 / 8.2 worst-seat bars, so K2 fails and K1 fires; reaching 5.95 would need about 46% of the
drafter's misses removed. The packages below spend engineering days only. The first card window that would capture data or train (W-T2,
9-18 window-hours) needs the owner's explicit K2 override at gate G, recorded.

## What is built (all on CPU, fakes and tiny models; allowlisted in qwen-integration-cpu.yml)

* WP-F1, the data pipeline: `ft_dataset.py` (example schema, bf16 feature files with a memory-mapped loader and sha256, the tap-major
  assembly of the four chips' 1,280-column slices, the storage arithmetic: 118 MB raw / 23.6 MB post-fc per 35k agent turn),
  `ft_anchor.py` (the recipe's anchor sampling and dense block mask, bit-exact against the recipe's own source where it is
  available locally), `ft_split.py` (refuses any overlap with A1, the eval tier or A2, and refuses own traces before D4),
  `ft_select.py` (writes a tau-lab data directory that `c2_tau_lab.py` drives as the new optional arm G, answers kept).
* WP-F2, the training reference: `dflash2_torch.py` (plain-torch DFlash2 with the checkpoint's tensor names), `ft_loss.py`,
  `ft_dflash2_train.py`, `ft_train_loop.py` (AdamW, clip, exact resume, export, a two-rank gloo run equal to the single process). This is
  the golden reference the ttml port is compared against.
* WP-F3, the ttml feasibility probe: `ttml_probe.py` (stages P0-P6, the decision rule, counts-only report; the REAL adapter is written
  at the first card window), `docker/tt-train/Dockerfile`, `patches/tt-train/backports.json` (the six fixes to carry; none fetched).

## Two findings from reading the sources

1. The block rule differs between training and serving. For a sliding-window draft the recipe's training mask is causal INSIDE a
   block (row k sees block rows 0..k); the inference model (z-lab, `is_causal` false) lets block rows see each other both ways.
   `ft_anchor.dense_mask` and `ft_dflash2_train.train_forward` take the rule as an argument (`specforge` / `full`) and a test pins that
   one anchor under `full` equals the inference forward and under `specforge` does not. Whether the serving drafter (SGLang, and our
   Tenstorrent port) reads the block bidirectionally is to be confirmed before any training choice rests on it.
2. About half of the A0 screen's code is the fine-tune's offline evaluation: the teacher-forced walk and the paired report.

## What is NOT done, and why

* `ft_render.py` (the thinking-ON renderer the lab's data builder used): its location is unknown to this work; the data builder is
  private. Until it is found, ids come from the lab's existing data directories.
* The feature-dump hook in S2 (P1) and its exactness gate: needs the serving stack and a card window.
* The full-size CPU parity check against the released drafter weights (10+ GB of RAM): needs a machine the owner names.
* The six upstream backports, the training image build, ttml on the simulator (G0), and probe stages P2-P6: need the rig image build
  and the cards; the Dockerfile and the manifest are written so a missing patch stops the build.
* A new job action for the probe in `c2_serving_job.py` and the hardware workflow: deliberately not added; it changes a hardware
  workflow that needs the runner allowlist.
* No new data intake: nothing was captured, nothing was read from the rig.

## Decisions for the owner

D-G record the K2 override before any W-T2 window; D4 the provider-terms decision before own traces are used as training contexts
(until then the mix is 50 / 30 / 20 SWE / chained / code, which needs none).
