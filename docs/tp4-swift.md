# Another checkpoint of the same architecture: Swift-1.5 on the four-card stack (tp4/swift)

The owner decided on 2026-10-12 to run `ukisai/Swift-1.5-Qwen3.8-27b` (revision `b4c84d42903a8646b25857eb2827288d92ed85a4`), an RL post-train of `Qwen/Qwen3.8-27B`,
on the TT stack. Its `config.json`, `generation_config.json`, `tokenizer_config.json`, `chat_template.jinja`, `tokenizer.json`, `vocab.json`, `merges.txt` and the two
preprocessor configs are byte-identical to the base's, its safetensors index names the same 1,199 tensors in the same 18 shards of the same sizes, and every weight shard differs.
That makes it a weights swap, not a port. `scripts/ci/qwen_c2_checkpoints.json` records the base's digests, Swift's, and the shard sizes and sha256 (public facts of the two
model repositories); the loader holds every listed checkpoint to the base's identity (`qwen_c2_checkpoint.load_catalog`, `test_qwen_c2_checkpoint`).

## How the stack chose its checkpoint before this branch

| what | where | effect |
| --- | --- | --- |
| the snapshot | every profile's `snapshots` (two roots: `/models/...` and `/models/hub/...`) in `qwen_c2_profiles.json`, resolved by `serving_c2_contract.resolve_snapshot` | `"@SNAPSHOT@"` in the engine argv (`--model`) and in `qwen_fast_runtime.target_snapshot` (`serving_startup` reads its `generation_config.json`) |
| the model the platform names | `--model Qwen/Qwen3.8-27B` in the platform argv, the smoke steps of `qwen-c2-serving.yml`, `c2_prefix_gate.PLATFORM_ARGS` | the contract OWNS the engine's model: `OWNED_FLAGS` drops the platform's `--model` and writes the snapshot, so the platform's id does not choose the weights |
| the served name | `--served-model-name` (the platform's), `c2_serving_gate.SERVED_NAME`, `prefix_replay.SERVED_NAME`, the replay's `SERVED_MODEL = Qwen/Qwen3.8-27B:tt` (the thin layer's alias) | names only; kept by the contract |
| the converted weights | `TT_CACHE_PATH` (the image's ENV, `/experiment-cache/weights`, a link into the hub's `.qwen-c2`), then the tt-metal model args add the device name and `weight_cache_path` (`docker/qwen-c2-graft/graft/model_config.py`) the dtype: `<TT_CACHE_PATH>/P150x4/tensor_cache_bfp8/<tensor>.tensorbin` | one file per tensor, **named by the tensor and nothing else**: a second checkpoint booted on it loads the first one's converted weights without a word |
| the kernel cache | `TT_METAL_CACHE` | compiled programs; independent of the weights, shared |
| pins of the base snapshot sha | `c2_serving_gate.SNAPSHOT`, `lever_n_*_gate.py`, `longctx_*`, `serving_canary_runner.py`, hardware scripts and workflows (`.../snapshots/1d4bf0f2...`), `full_mtp_request.py`, `dspark-target-hardware.py`, `t32-combined-hardware.py` | research and gate harnesses, not the c2 smoke/tau path; they keep the base and are out of this change |
| the drafter | `/draft-config`, `/experiment-dflash-fixture`, `drafter_checkpoints.json`, manifests | pinned to its own checkpoint (manifest `dedf8df6`); the manifest holds no embedding or LM-head tensor, so those come from the target at run time and follow it |

No evidence record (packed-any, W2, engine reuse, fusion audits) pins the base's weights: they pin source files and the runtime binary. They were made on the base's weights, so
for Swift they are an argument (same code, same shapes), and SA1 of the pack is the card check.

## How a checkpoint is selected now (default off)

* Job keys `C2_MODEL_REPO` + `C2_MODEL_REVISION` (`c2_serving_job.read_model`): both or neither, a pair `qwen_c2_checkpoints.json` lists, not the base, the 40-hex revision. Only the
  smoke and tau lab steps forward them (`status rescan reset smoke taulab`, four cards); every other step is refused. The workflow starts the container with
  `QWEN_C2_MODEL_REPO` / `QWEN_C2_MODEL_REVISION`, `--model <repo>`, and the checkpoint's own served name; `c2_smoke_check.py --expect-checkpoint` and the tau lab's launched-argv check read the
  container log (a run that asked for Swift and shows no checkpoint line fails; a run that asked for nothing and shows one fails).
* The same two names are the serving environment (`serving_c2_contract.configure_checkpoint`, called from `boot()` in every process). In the API server: the snapshot is the profile's
  own pin with the base's hub directory replaced (`/models` and `/models/hub` both work); every identity file must hash to the BASE's digest, the index's `weight_map` must be the base's, every
  shard must be there at its pinned size, no `*.incomplete` may be in the blob directory; `HF_MODEL` / `MODEL_WEIGHTS_DIR` (which the loader prefers over `--model`) must be unset or the snapshot;
  the platform's `--model` must name the model the image serves (`check_platform_model`: a platform asking for Swift on an image with no selection, or for the base on an image set to Swift, is
  refused, so a checkpoint is never answered under the other's name). Any refusal exits the API server (78).
* **The converted-weight cache is keyed by the pair**: `TT_CACHE_PATH` becomes `<TT_CACHE_PATH>-ckpt/<org>--<name>--<revision>` in every process (`QWEN_C2_WEIGHT_CACHE_BASE` keeps the base's so a
  child that inherits the keyed value derives the same directory). It is a sibling of the base's tree, never under it. `CHECKPOINT.json` beside the tensors records the pair and the weight_map digest; a
  directory whose marker names another pair is refused; the worker sets `first_boot_complete` after the fast path attached (a boot that died in the middle of the conversion leaves it false and the
  next boot says so).
* The worker re-checks after the model is built and before the fast path attaches (`serving_startup.bind_checkpoint`): the model's checkpoint directory is the snapshot, its weight cache is under the
  keyed directory, the engine's `target_snapshot` is the snapshot; else the worker stops. It logs `[QWEN-C2] checkpoint bound ...`.
* **Answer room (owner-approved 2026-10-12)**: the Swift catalog entry carries `"request": {"default_max_tokens": 16384}`; `serving_c2_contract.checkpoint_limits` lays it over the profile's 8,192 in the API
  server (the ship-4 profile and the base are untouched). A request that omits `max_tokens` gets 16,384, clamped by `enforce_request` to the room a long prompt leaves (a 253,920-token prompt gets 8,192) and
  by the output budget (16,384, unchanged); an explicit larger value is clamped as before. The per-request KV check never refuses a default request. Cost: the reservation holds each request's
  worst case (prompt + max_tokens), so with the default max_tokens eight seats fit the 16,499-block pool up to prompts of 115,488 tokens (123,680 with the 8,192 default); between those, the eighth
  default-max_tokens request waits for blocks instead of running. Requests that name max_tokens at 8,192 or less are unaffected.
* Proofs: `test_qwen_c2_checkpoint` (the base and Swift resolve to different cache directories; the base's path is exactly what it was, in the environment, `keyed_cache_dir` and `contract.boot`; the off
  path imports nothing; a snapshot whose config, tokenizer or template differ is refused field by field; the worker stops on the base's weights or cache), `test_swift_pack`.

## Served-model-name / alias plan (a proposal; the base's alias is not touched)

* One node serves one checkpoint (the four cards hold one model), so a Swift deployment replaces the base on its node; nothing switches per request.
* Placement model id (what `serving.manage --model` and the node's `served_models` carry): `ukisai/Swift-1.5-Qwen3.8-27b`, which is also what the platform passes as `--model`
  (the contract refuses a mismatch with the image's environment). Thin-layer alias (`THATCH_SERVING_MODEL_ALIAS`): `ukisai/Swift-1.5-Qwen3.8-27b:tt`, the same opt-in convention as the base's
  `Qwen/Qwen3.8-27B:tt`: the TT node never advertises the plain checkpoint id (other fleets may serve it), and agents opt in by naming the `:tt` id.
* No alias is shared between the two checkpoints, and `Qwen/Qwen3.8-27B:tt` is never pointed at Swift: a client that names the base must get the base or an error, never Swift. While a node serves Swift,
  `Qwen/Qwen3.8-27B:tt` is not advertised by it.
* The job windows already serve Swift under `ukisai/Swift-1.5-Qwen3.8-27b` (the `model` field of the answers says which model replied).
* To do before a production use (not in this branch): the thin layer's ENV lines (`QWEN_C2_MODEL_REPO`, `QWEN_C2_MODEL_REVISION`; the node agent forwards neither), the alias above, and the replay job
  (`c2_platform_replay.py`: its `--model` default and `CHECKPOINT` pin are the base's; `C2_REPLAY_SERVED_MODEL` exists, a replay model key does not).

## The window (`scripts/ci/references/fusion-jobs/SW`, `make_swift_jobs.py`; ORDER.txt has the minutes and the read rules)

B0 build, X0, **W0** first boot of Swift (builds its weight cache; box 150 min), **SW1** smoke (warmup, coding, solo prefill ladder to 254k, concurrent8_steady, the coding set), **SB1** the same on the
base, **T1/T2** the tau lab on the base and on Swift (the base's drafter, unpaired: `swift_tau_compare.py`), **C1 L1 C2 L2** the decode ABAB (concurrent8_steady and concurrent8_code_32k_long) on the
ship-4 profile, **SA1** the audited attach on Swift (exactness of the fast path on its weights), Z. About 11 h 45 min; cut it at a block end.

The coding set (`references/swift/coding-problems.json`, the smoke test `coding_set`, `swift_coding_report.py`): twenty original self-checking Python problems with hidden tests; each model answers
greedily with thinking on (12,288 tokens at most); the report runs the code on a disposable CPU host and prints passes, thinking and answer tokens per model and the pairwise table.

## Risks

* Greedy only: the model card's numbers are at temperature 1.0 and this stack's request contract serves greedy; RL post-trained models can loop or lengthen under greedy. The coding set reads it.
* Drafter fit: the DFlash drafter was fit to the base; tau on Swift is unknown until T2 (a lower tau is a slower decode).
* Conversion time of the first boot: the base's whole four-card converted cache (1,246 tensor files, 24 GB) has file times within 3 minutes 15 seconds, so the conversion is a few minutes; with a cold read of the 52 GB of
  safetensors and the warm boot the first boot is an ESTIMATED 8 to 25 minutes. The smoke step waits 80 minutes for readiness; a boot killed during the write leaves a partial cache directory (delete it and re-run W0).
  A production deploy must find the cache built: the platform's bounded load attempt is about 13.5 minutes. The keyed directory is on the hub (`/experiment-cache` is a link to the hub's `.qwen-c2`, which the card-job container
  and the platform container both mount read-write), so the cache a card job builds is the one the platform boots on.
* Disk: the hub filesystem must take tens of GB more for the converted weights, and the window image is another 30 GB.
* Licence: the Swift repository carries its own licence (see its LICENSE and NOTICE; commercial use is conditional); that is the owner's decision, not the stack's.
* Evidence: every exactness record was made on the base's weights; every Swift number is UNQUALIFIED.
