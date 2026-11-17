# tp4/s6-notier: the ship-6 profile without the host KV tier

Branch `tp4/s6-notier`, cut from `tp4/s6` at `43c60a52` (the ship-6 waiver approved). Owner decision, 2026-11-17: take the host KV tier off production,
keep the 16 GiB checkpoint store, trim the heap once after warm-up, and boot without the vision tower when the profile serves no images. The goal is about
32 GiB of host RAM back on the card host, which CI shares.

## 1. Why

What the engine process of the ship-6 image (`-fx-traffic-s6`, waiver `ship6-fx-262k-2026-10-21`) holds, from the 2026-11-17 measurement of the live engine:

| Piece | Size | What it is |
| --- | --- | --- |
| shared anonymous mapping | 32 GiB | The host KV tier's slab: `QWEN_PREFIX_HOST_TIER_GIB=48` less `QWEN_PREFIX_STORE_GIB=16`. `TierStore.configure` maps it once (`mmap.mmap(-1, n)`). It never shrinks: eviction only refills a Python free list, and the kill switch only clears records. |
| heap | about 22 GiB | About 16 GiB is the GDN checkpoint store at its budget. The rest is runtime and allocator retention. |
| anonymous | about 4 GiB | UMD sysmem, 1 GiB per P150. It cannot be freed. |

The engine's own counters (`/tmp/qwen-prefix-stats.json`) after five days of traffic:

| Counter | Value |
| --- | --- |
| `tier_bytes` | 32 GiB (the cap) |
| `tier_spill_blocks` | 90,950 |
| `tier_restore_requests` / `tier_restore_blocks` | **0 / 0** |
| store bytes | 15.88 of 16 GiB |
| `class_returning_served` | 4,394 |
| `class_kv_evicted` / `class_ckpt_evicted` | 1 / 4 |

The tier spilled 90,950 blocks and restored none. Returning sessions were served from the device pool (16,500 blocks) and the checkpoints. So removing the
tier frees 32 GiB, and the measured cost is at most the one returning session whose KV left the device pool.

## 2. What changes

| Item | Where | Default | In the new profile |
| --- | --- | --- | --- |
| New traffic profile `c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s6-nt` (62 characters; a baked default may have 64) | `scripts/ci/fusion-wp/S6NT.json`, generated into `qwen_c2_profiles.json` by `make_fusion_profiles.py` | - | - |
| Host KV tier removed: `QWEN_PREFIX_HOST_TIER_GIB`, `_AUDIT`, `_VERIFY`, `_SPILL_MODE`, `QWEN_PREFIX_TIER_RESTORE_SLICE`, `_MS` | the ship entry's env | tier off is the registry default | absent |
| Checkpoint store | the ship entry's env | - | `QWEN_PREFIX_STORE_GIB=16` (unchanged) |
| Owner traffic waiver `ship6-notier-2026-11-17` | the ship entry's waiver | - | the ship-6 levers less `QWEN_PREFIX_HOST_TIER_GIB` and `QWEN_PREFIX_HOST_TIER_SPILL_MODE`; the ship-6 text and evidence verbatim, then (S) and (T) |
| Text-only boot, `QWEN_C2_TEXT_ONLY_BOOT=1` | `serving_c2_contract.py` item 10 (`text_only_problems`, `skip_vision_init`, `install_text_only_boot`) | **off** | on |
| `malloc_trim(0)` once after the fast worker's warm | `serving_startup.trim_host_heap`, called from `warmup` after `start` | on in every profile of the image, glibc only, never fatal | on |

Nothing else of ship 6 moves. The ship-6 profile and its waiver are untouched. The rollback is the image that serves them now, not that profile on
this branch's image (section 4). The file's own default (`c2-packed-tp4`) is untouched; the build bakes the profile (`C2_BAKE_DEFAULT_PROFILE`).

The ship entry also names the checkpoint production serves beside the profile (`"model"`: Swift, `ukisai/Swift-1.5-Qwen3.8-27b` at `b4c84d42`). It
changes no profile. It puts the checkpoint selection on the pack's smoke and prefix gates (section 4).

### The contract

The contract already admits a traffic profile with no tier:

* `prefix_policy_problems` checks the tier's numbers, the sliced restore and the spill mode only when they are present.
* The waiver may name only the levers the profile sets. So the two tier levers leave the waiver with the env lines.

No tier rule changed. The new rule is for the text-only switch only: `0` or `1`, and `1` only when the engine's `limit-mm-per-prompt` is exactly
`{"image": 0, "video": 0}`. An inherited value is dropped under any profile that does not name the switch.

### The text-only boot

The serving wrapper (`qwen36_vllm.initialize_vllm_model`) calls `model.init_vision_model()` with no arguments, unconditionally. That call loads the whole
checkpoint a second time (`reference_vision_model`, a `from_pretrained` of the VLM) to keep its visual half, builds the vision weights on the device, and keeps
the HF tower alive, which pins one safetensors shard's mapping.

This profile cannot serve an image anyway:

* the engine admits image 0 and video 0;
* the wrapper asserts that batched (`max_num_seqs > 1`) serving is text-only.

With the switch on, every process wraps `Qwen36Model.init_vision_model` when `models.demos.blackhole.qwen36.tt.model` is imported:

* the default call (no arguments, no tower yet) returns `None` and logs `[PINDIAG] text-only boot: vision tower skipped` once;
* every other call runs the original.

The wrapper's file is not edited: it is a prefix-stage target with pinned bytes.

It is behind a switch because nothing of it has run on a card. The device DRAM layout at the attach changes: the vision weights are no longer there, so
more DRAM is free. It is the one switchable line of item (T): delete `QWEN_C2_TEXT_ONLY_BOOT` from the ship entry of `S6NT.json` and regenerate.

### malloc_trim

* `trim_host_heap` runs once, after `start(worker)` returns and before the scheduler exists. Nothing in the image called `malloc_trim` before.
* It loads `libc.so.6` only when `os.confstr('CS_GNU_LIBC_VERSION')` reports glibc.
* It calls `malloc_trim(0)` and logs VmRSS before and after on one `[PINDIAG] host heap trimmed after warmup` line.
* It never raises.
* There is no periodic trim. A trim walks every arena under its lock, and the heap grows to about 22 GiB under traffic, so a timer trim would hold the
  allocator for a time nobody has measured.
* It frees only what the load and the captures left free. The checkpoint store's 16 GiB stays in use by design.

## 3. Expected host RSS

| | Ship 6 (measured) | This profile (estimate) |
| --- | --- | --- |
| tier slab | 32 GiB | 0 |
| checkpoint store (heap) | about 16 GiB at budget | about 16 GiB at budget |
| UMD sysmem | about 4 GiB | about 4 GiB |
| runtime, fragmentation, drafter fixtures, trace copies | about 6-10 GiB | about 5-9 GiB (the trim) |
| **process total** | **about 60 GiB** | **about 25-29 GiB** |

The text-only boot changes host RSS by about 0-1 GiB. Its gain is a faster, lighter boot: no second checkpoint load and no vision weights.

## 4. The gates and the build

The generated pack is `scripts/ci/references/fusion-jobs/ship-traffic-s6-nt/`. It holds B0, G1, GX0, GX1, G2 and SR on the placeholder image
`tp4-s6nt-1`. The control of GX0 is the ship-6 profile.

### Which checkpoint each job serves

Production serves Swift. Three ENV lines of the production thin layer select it: `QWEN_C2_MODEL_REPO`, `QWEN_C2_MODEL_REVISION` and the alias
`THATCH_SERVING_MODEL_ALIAS=ukisai/Swift-1.5-Qwen3.8-27b:tt`. The node agent forwards no environment into the container.

| Job | Checkpoint | Why |
| --- | --- | --- |
| B0 | none | A build. |
| G1, GX0, GX1, G2 | Swift, by `C2_MODEL_REPO` and `C2_MODEL_REVISION` | They run the profile as production boots it: Swift's own converted-weight directory, and Swift's catalog request numbers laid over the profile's (`checkpoint_limits`: a default answer of 16,384 and an output ceiling of 32,768, ship-6 waiver item K). Both GX arms carry the same keys, so the pair differs by the profile alone, as in the S6 window. |
| SR | the base | The replay step cannot carry a selection: the job reader refuses the keys beside it (`MODEL_ACTIONS`), the workflow passes no `--model`, and the replay's served-name check defaults to the base's alias. On the production layer the contract refuses the base model at every load (`check_platform_model`, exit 78). |

So SR runs on a **base variant** of the thin layer:

* the same build as the production layer, without the three Swift lines;
* build the production layer as the base variant plus one layer of exactly those three lines, so the sorted `Config.Env` of the two images differ by
  exactly them;
* SR qualifies the platform sequence and the B0 bytes on the base checkpoint, not the production bytes;
* the replay copies the tracked record of the base's placement (`references/c2-serving`), so the source image that record names must be on the rig
  host first;
* the base's own converted weights must be on the hub; a cold cache is converted at the first load and can outlast the replay's startup wait.

The production layer is first served by the library deploy (`release_first`). Read after the deploy:

* `/v1/models` names `ukisai/Swift-1.5-Qwen3.8-27b:tt`;
* a tool call comes back as `tool_calls`, and the reasoning is split from the answer;
* the no-tier read below holds in the engine log.

### What must pass

The pass rule is in the pack's `ORDER.txt`. Before any publish, placement or `:latest` move, all of these must pass:

* B0;
* G1 (the eight-seat 262k gate);
* GX0 then GX1 (exactness: the same prompts on both profiles). This pair is the only exactness evidence for the text-only boot and the changed DRAM
  layout;
* G2 (the salted prefix hit; soft, or the owner's recorded override against its control);
* SR, which needs G1 and GX1.

A smaller gate depth is the owner's decision. Record it in `ORDER.txt` before B0.

### The decision

The waiver's decision is **PENDING**. The owner's decision of 2026-11-17 reached this branch through another session, so the sentence records it as
relayed and waits for the owner. B0 is refused while the decision is PENDING, by `c2_serving_job.read_bake` and `build-c2-serving-image.sh`.

The owner approves with one sentence starting APPROVED. Commit it, run the CPU suite on that commit, and push the B0 tag on that commit:

```
python3 scripts/ci/make_fusion_profiles.py --approve-ship "APPROVED ..." --ship traffic-s6-nt
```

### Rolling back

Roll back by image, never by profile. The rollback is the image that served before this one.

* Selecting the ship-6 profile on this branch's image is not ship 6. The post-warm `malloc_trim` runs in every profile of the image, and it has no
  switch. The GX0 control trims too. The trim cannot change an answer, but its effect on host RSS cannot be measured A/B on one image.
* The rollback image must be in the registry before the deploy. Check its manifest. A tag that exists only on the card host dies in the next prune.

Read rules beyond the pack's own:

* no `host tier on` line;
* the text-only line once per engine process;
* the heap-trim line once;
* every ship-6 line unchanged: the governor's install line, and WC's engaged line with its one octo decline.
