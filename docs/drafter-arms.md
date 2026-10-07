# Drafter arms: which DFlash2 drafter serves, and how a candidate is judged

Branch `tp4/drafter-arms` (off `tp4/levern-prefix`). Everything here is behind a default-off selector: the production profile, the default
manifest and the served drafter bytes are unchanged. Nothing in this branch runs on a card until the combined window.

## Design

**The question.** The served drafter is `incoai/Qwen3.8-27B-DFlash2` at the pinned revision (short form `dedf8df6`), trained at block size 8 and run
by us at T16 (15 proposals). Two cheap weights-only candidates exist, and neither changes a shape, a kernel or the tensor-parallel layout:

- `0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16` (revision short form `98759a49`): a public fine-tune of the served weights at block size 16, which
  is exactly our T16. Apache-2.0 with a LICENSE file in the repository. Its safetensors header is byte-identical to the served drafter's (the same
  81 BF16 tensors, shapes, offsets and total size); its config differs only in `block_size`. We hashed all 81 tensors from a full download of the
  pinned revision: 70 differ from the served drafter (all five layers' projections and convolutions, the `fc` feature projection and the norms), the
  three candidate-selector tensors and eleven norm vectors do not. The LFS object hash of the file matches the hub's. Its published results are
  thinking-off, 256-token chat, one request at a time; nothing it reports says what it does to tau on thinking-on coding-agent turns at 25-66k
  prompts. That is what the tau lab measures.
- `JonasLoos/Qwen3.8-27B-DFlash2-b32`: **excluded**. Its licence is a metadata field only (no LICENSE file), part of its training prompts have
  an unstated licence, and serving refuses anything but 15 proposals, so its positions 16-31 could not be used anyway.

A third arm needs no new weights: the served drafter with `QWEN_FAST_DRAFTER_BF16=1` (all 36 projection uploads in bfloat16 instead of the
image-baked bfloat8_b; +405 MB per chip). That flag has never run on hardware.

A drafter cannot change the verified text, only the committed tokens per round (tau). So the bar is statistical and paired: see the GO rule.

**Manifests, not edits.** The drafter's identity (repository, full revision, safetensors header hash, byte size, the 81 per-tensor sha256 values,
the hash of the first 32 rows of `fc`, the config hash) is data under `scripts/ci/references/drafter-manifests/<name>.json`. The loaders that pin the served
drafter (`draft_*_fixture.py`, `full_dflash_request.py`), the simulator evidence that embeds its manifest and the frozen evidence that hashes those
sources are **not edited**: any change to them risks the pins that qualify the production image, and there is no way to check the frozen set from a CPU
suite. Instead:

- `drafter_manifest.py` selects a manifest: `QWEN_DRAFTER_MANIFEST` first, then the `DRAFTER_MANIFEST` marker file the image build lays beside the
  fixtures, then the default. The default manifest (`dedf8df6`) is pinned here by the sha256 of its canonical content, and
  `test_drafter_manifest` holds every value in it equal to the unedited loaders' own pin tables.
- `drafter_fixtures.load` is what `serving_startup` now calls. The default manifest goes straight to the unedited `load_dflash_fixtures`; a
  candidate manifest goes down a parallel path that repeats the same checks (manifest identity, per-file size, shape, dtype and hash, the finite check, the
  `fc` slice) against the candidate's pins and returns the same four values. A candidate image whose fixtures are another revision's refuses at attach, and a
  candidate prints one `[DRAFTER_MANIFEST] <name> in force` line (the default prints nothing new).
- The image build (`build-c2-serving-image.sh`) takes `C2_DRAFTER_MANIFEST` (job key `C2_DRAFTER_MANIFEST`, build action only): the candidate's
  model and full revision come from its manifest, the draft config is fetched at that revision and refused unless its sha256 is the manifest's, and the marker is
  laid. Unset, the script does what it always did (its default also now checks the config hash, which is the one in the default manifest; a test holds the script's
  three repeated values equal to the default manifest's). A candidate's fixtures are read from their own staging cache (`C2_DRAFTER_FIXTURES`, a plain path under the
  build user's home; never the served drafter's cache), and a missing cache stops the build at once with the `drafter_stage.py stage` command to run: that arm is then lost,
  and the pack's ORDER says so. A candidate image is built and smoked but **never pushed**: the job reader refuses `build push` with a candidate manifest, and the push
  step refuses any image that carries the `DRAFTER_MANIFEST` marker.
- The two overlay manifests are the first overlay files below `/experiment-scripts/ci`, a directory the P8 base lacks, so the Dockerfile makes
  `/experiment-scripts/ci/references/drafter-manifests` before the overlay installs (the installer refuses a destination whose directory is missing). A test stages the real
  overlay and installs it into a base-shaped tree. The fast-serving (P8) image copies `serving_startup.py`, which imports `drafter_fixtures`, so both fast-serving copy
  lists carry `drafter_fixtures.py`, `drafter_manifest.py` and the manifests directory; the copy-closure test now also reads plain `import` statements.
- Both hub `config.json` files are committed (`references/drafter-configs/`, hashes equal to the manifests') with a test that they differ in `dflash_config.block_size`
  only, and a real-vLLM test (`test_drafter_config_vllm`, run by `qwen-fast-vllm-cpu.yml`) that builds vLLM's dflash `SpeculativeConfig` from each with the profile's
  arguments and holds the scheduler-side values equal.
- `drafter_stage.py` hashes and splits a downloaded checkpoint into the fixture layout (`describe` prints a manifest draft; `stage` writes the
  directories after checking every byte against a committed manifest). It reads a local file, so the 3.85 GB download happens on the build host, never a
  laptop. The b16 manifest in this branch was produced that way.

**Tau lab, drafter mode.** The W-T1 tau lab (`c2_tau_lab.py`, `tau_lab_report.py`, the `taulab` action of `qwen-c2-serving.yml`) is ported from
`tp4/tau-lab`. New: `--drafter-arm` (job key `C2_TAULAB_DRAFTER_ARM`) with three arms, selected by the image (its tag carries the manifest's fixtures) and by the derived
profile, which adds only that arm's drafter-only flags on top of the production profile and the lab's two log flags:

| arm | image | flags added | log marker the lab requires (it refuses and exits 2 before sending anything without it) |
|---|---|---|---|
| `control` | default manifest | none | none |
| `b16-bf8` | candidate manifest | `QWEN_DRAFTER_MANIFEST` | `[DRAFTER_MANIFEST] ... in force` |
| `dedf-bf16` | default manifest | `QWEN_FAST_DRAFTER_BF16=1` | `[DRAFTER_BF16] engaged` |

The drafter flags move the proposals and never the text, so they are the one arithmetic difference an arm may add (the lab's check refuses any
other). The refusal is real: with a drafter arm, a container log that does not show the arm's own markers (the wrong image tag, a bf16 switch that never engaged) ends the lab before
the first request and records `error: ArmMismatch` in the run. A3 (the calibration, which must reproduce the served drafter's own reference within 7%) runs on the control arm
**only**, and **first** on it (whatever order the job lists the arms), and the lab stops after it on a FAIL: a candidate that moves
tau would fail it by construction, so the default arm list leaves it out of a candidate and a list that names it is refused. For a drafter arm the 80k+ turns are
moved forward (every third turn, in the shuffled order both arms share), so a short arm still reaches the three paired 80k+ turns the long-context clause needs. The production label was stale
(it required the old image and the verify audits on); it now knows each production image's own audit setting. Prompts are the public SWE-rebench held-out set (A1)
and the scrubbed own-session set (A2) the lab already reads from its rig-local data directory, thinking ON; no customer traffic.

**The pair report.** `drafter_pair_report.py` reads a control run and a candidate run (both private, rig-local; the lab's `taulab` step runs it when
`C2_TAULAB_PAIR_CONTROL` names the control run) and pairs them by turn id.

GO rule, pre-registered, every clause required:

Every ratio below is over the paired turns whose text was the same in both arms; a diverged turn is counted by clause 4 only.

1. **pooled**: the ratio candidate/control of the equal-weight-per-set pooled tau (A1, A2, A4) over paired turns has a 95% cluster-bootstrap lower bound above 1.00.
2. **long**: no regression in the 80k+ bucket: point ratio at least 0.98 and 95% upper bound at least 1.00.
3. **p10**: the same two conditions on the ratio of the arms' per-turn p10 (turns with at least 8 counted rounds in both).
4. **text**: every paired turn has identical output token ids in both arms (the exactness contract); a difference is a NO-GO until explained.
5. **coverage**: paired turns are at least 90% of the control's.
6. **arms**: the control run's drafter arm is exactly `control` and the candidate's is the arm asked for.
7. **launch**: neither run recorded a launched-container problem or an error.
8. **matched**: the two runs agree on profile, seed, max_tokens and per-arm data counts, and their image tags differ for `b16-bf8` and are equal for `dedf-bf16`.

Any FAIL is NO-GO; no FAIL with something unestablished is NOT_ESTABLISHED; otherwise GO. The control's A3 verdict is reported beside the verdict
(`control_calibration`) and is **not** a GO precondition: it compares eight turns with an older stack's reference, and the paired ratio is a within-run comparison.
The job reader also refuses a candidate arm that names no `C2_TAULAB_PAIR_CONTROL`, so a forgotten edit cannot leave a window without a verdict. The summary holds aggregates only (the report's public-words check).

**Card gates** (`scripts/ci/references/tp4-drafter-jobs/`, parameterised templates; no private detail): X0 status/rescan, B0 and B1 builds (control image and
the candidate image from the same commit), the three tau arms D-T1 (control, with A3 first), D-T2 (b16-bf8), D-T3 (dedf-bf16), 80-85 min deadlines each (about 4.3 h for the three),
and, **for a GO arm only**, the qualification (about 4.8 h, sized from the W1 window's measured job times; if it does not fit the combined window it moves to a follow-on): S1 audited smoke (8 x 262k, every audit incl. the draft singles and quad audits), bringup + matrix against the tracked
v235 texts, the singles/quad audits on the four-live paths, five consecutive hang-shape completions, the 8 x 262k attach, a short ABAB against the control image, and Z.
Kill rules are in each template and `ORDER.txt` (machine-greppable `# NEEDS` lines). **A bf16 GO has no qualification path in this pack**: no gate profile
pairs the bf16 weights with the production 8 x 262k prefix shape (the bf16 pool is 19,200 blocks, not 19,968), so the follow-on needs `c2-packed-tp4-8x262k-ship-prefix-dbf16`
and an audit twin first; ORDER says so.

## Files

- `scripts/ci/drafter_manifest.py`, `drafter_fixtures.py`, `drafter_stage.py`, `references/drafter-manifests/{dedf8df6,b16-98759a49}.json`, `references/drafter-configs/`, `test_drafter_manifest.py`, `test_drafter_config_vllm.py`
- `scripts/ci/c2_tau_lab.py`, `tau_lab_report.py`, `drafter_pair_report.py`, `test_tau_lab.py` (ported), `test_drafter_arms.py`, `references/tau-lab/`, `references/tp4-taulab-jobs/`
- `scripts/ci/references/tp4-drafter-jobs/` (the card-gate pack); `references/tp4-taulab-jobs/` is the earlier W-T1 pack, marked superseded and not run again
- not changed on purpose: `dflash-fixtures.sh`, the helper the older lever-N gate arms use to fetch the served drafter's fixtures (a bundle file; the image build and the tau lab take a candidate through `build-c2-serving-image.sh` instead).
- changed: `serving_startup.py` (one call), `build-c2-serving-image.sh`, `c2_serving_job.py`, `qwen-c2-serving.yml`, `qwen-integration-cpu.yml`, `qwen-fast-vllm-cpu.yml`, `qwen-fast-serving-image.yml`, `docker/qwen-c2-overlay.txt`, `docker/qwen-c2-serving.Dockerfile`, `docker/qwen-fast-serving.Dockerfile`

## Open items

- **Does vLLM's DFlash path read `block_size` from the draft config?** Read from the pinned vLLM's source: no.
  Its DFlash code reads `dflash_config`'s `causal`, `use_swa`, `swa_window_size`, `mask_token_id`, `target_layer_ids`, `use_aux_hidden_state` and the sink-bias flag, and nothing asks it
  for `block_size` (the `block_size` its proposer uses is the KV page size). No TT source reads `dflash_config` either (only `dspark_intake.py` mentions it). `test_drafter_config_vllm`
  holds that against the installed vLLM on every run: the two configs load equal but for the key, and no vLLM module reads it. What a card could still show is the candidate's
  acceptance at T16, which is the tau lab's question.
- **The lab's statistic counts rounds with all four seats live**, so the drafter arms run the four-seat serving profile (`c2-packed-tp4`), as the lab was validated. An eight-seat
  statistic needs the report's live rule generalised; tau per round per seat does not depend on the seat count, so this does not bias the ratio.
- **A3's reference** is from the earlier stack; the control arm re-calibrates against it on the current image. If it fails by a stack effect rather than a drafter effect, re-derive the reference from a control run first.
- **bf16 memory at 8 x 262k**: the bf16 weights need a 19,200-block pool (the existing dbf16 twin); the tau arm runs at four seats where it fits, and the timing question is the dbf16 pack's.
- **Provenance of a candidate base image**: the candidate image is built by the integration step; its fixtures must be staged on the build host first, and a G1 run on a candidate image has not been exercised.
- **Frozen evidence not checked**: we could not enumerate which sources the baked frozen evidence pins, which is why no pinned loader is edited. If the integration build's G1 reports an overlaid source breaking a pin, it is one of the files this branch adds or changes in the overlay list (`serving_startup.py` is documented as unpinned).
