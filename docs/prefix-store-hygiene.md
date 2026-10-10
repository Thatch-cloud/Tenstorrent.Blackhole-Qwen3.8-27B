# Prefix checkpoint store: telemetry and hygiene

The GatedDeltaNet (GDN) checkpoints that make a prefix hit exact live in `qwen_prefix_registry.PrefixRegistry`, one per
engine process. This note covers what that store now measures about its misses, the two opt-in policies that keep it
useful, the size arithmetic behind its budget, and two opt-in features that make a returning turn cheaper: checkpoints
stored in the device's own layout, and a host-RAM tier for the attention-KV pages vLLM evicts. Nothing here changes what a
hit serves: a checkpoint is still token-verified before it is granted, a retired or evicted checkpoint is only ever a miss,
and everything restored from host RAM is, byte for byte, what was evicted.

## Checkpoint size

One checkpoint is the GDN state of the whole mesh after a 2048-token boundary: 48 layers of recurrent state
`[48 value heads, 128, 128]` plus a bf16 conv carry `[3, 10240]`. The totals do not depend on how many chips share them.

| Recurrent state | Bytes per checkpoint | Checkpoints in the default 8 GiB store |
|---|---|---|
| bf16 (`QWEN35_GDN_STATE_BF16=1`, the serving setting) | 78,446,592 | 109 |
| fp32 (the model default) | 153,944,064 | 55 |

The model reports the real size of what it reads; `qwen_prefix_registry.checkpoint_nbytes()` derives the same number
from the state dtype for the registry's default charge, Lever N's host-memory guard (`levern_policy.park_nbytes`, and
`observed_park_nbytes`, which prefers the size of the last scratch the route really read) and the gate's capacity
arithmetic (`prefix_judge.store_entries(..., state_bf16=True)`). A gate that judges a store of a given size must be told
which dtype the engine ran.

## Admission classes (always on; `QWEN_PREFIX_TELEMETRY=0` turns the whole of it off)

Every admission is classified once, when the step admits it, by what the trim answered (vLLM's raw hit `h`, the trimmed
`Q`) and by a bounded memory of earlier turns. A turn is remembered under the block hash at its last boundary (the prompt
boundary `floor2048(P)`, less one chunk under sticky sessions); a later request whose own hash chain contains that hash is
a *returning session*, and the highest such boundary is the one it should have resumed at.

| Class | Meaning |
|---|---|
| `returning_served` | a returning session whose trimmed hit reached its previous boundary |
| `first_turn` | no earlier turn, and its first block (which carries the salt) was never seen |
| `rewritten` | its first block was seen, but no earlier turn's boundary is a prefix: another session's shared prefix, or a rewritten, compacted or edited history |
| `short` | too short to hold a reusable boundary |
| `kv_evicted` | returning; vLLM's hit stopped below the previous boundary (the device pool evicted the blocks) |
| `ckpt_evicted` | returning; the KV survived but the checkpoint was stored and then removed (`class_ckpt_evicted_<why>`: `budget`, `coupled`, `superseded`, `cleared`, `other`) |
| `ckpt_missing` | returning; the KV survived and no checkpoint was ever stored there (a capture failed or was never planned) |
| `refused` | returning; the checkpoint is resident but was not granted (`class_refused_mismatch` counts token-id mismatches) |
| `unsalted` | no `cache_salt`: no hit is possible |
| `denied` | a streaming-input session, or the kill switch is latched |

The classes partition the admissions. `class_<name>_hit` counts those of a miss class that still got a partial hit
(through a prefix shared with another session, or an older boundary), and `lost_tokens_<class>` sums the tokens a
returning session had to prefill beyond what a full hit would have saved.

A returning session's **reuse distance** goes into two histograms, split by outcome (`served` or `missed`): the seconds
since its previous turn, and the prompt tokens admitted for *other* requests since then (the span a cache would have had
to hold across).

### Prometheus (the engine's `/metrics`)

| Name | Kind |
|---|---|
| `qwen_prefix_class_<name>_total`, `qwen_prefix_class_<name>_hit_total`, `qwen_prefix_class_ckpt_evicted_<reason>_total`, `qwen_prefix_class_refused_mismatch_total` | counters |
| `qwen_prefix_lost_tokens_{kv_evicted,ckpt_evicted,ckpt_missing,refused}_total`, `qwen_prefix_returning_sessions_total`, `qwen_prefix_admitted_prompt_tokens_total`, `qwen_prefix_class_failures_total` | counters |
| `qwen_prefix_reuse_seconds{outcome}`, `qwen_prefix_reuse_tokens{outcome}` | histograms |
| `qwen_prefix_host_rss_bytes` (the engine process), `qwen_prefix_host_mem_available_bytes` (the host's `MemAvailable`) | gauges |
| `qwen_prefix_registry_bytes`, `qwen_prefix_registry_entries`, `qwen_prefix_registry_budget_bytes` | gauges (as before) |
| `qwen_prefix_ghost_now`, `qwen_prefix_evict_fair`, `qwen_prefix_supersede`, `qwen_prefix_telemetry` | gauges: the memory's size and the policies in force |

Metrics are per class, never labelled by tenant.

### Log lines

At the stats cadence (`QWEN_PREFIX_STATS_S`, default 30 s), whether or not a counter moved, the engine log gains two
`key=value` lines beside the existing `[PINDIAG] prefix: stats {...}` JSON:

```
[PINDIAG] prefix: tier rss=<bytes> avail=<bytes> reg_bytes=<n> reg_entries=<n> reg_budget=<n> ghost=<n> adm=<n> adm_tokens=<n> grants=<n> returning=<n> evict=<lru|fair> supersede=<0|1> returning_served=<n> first_turn=<n> ... denied=<n> lost_kv=<n> lost_ckpt=<n> lost_missing=<n> lost_refused=<n> superseded=<n> evicted_lru=<n> evicted_fair=<n> evicted_coupled=<n>
[PINDIAG] prefix: reuse bounds_s=<csv> served_s=<counts csv>:<sum> missed_s=<counts csv>:<sum> bounds_tok=<csv> served_tok=<counts csv>:<sum> missed_tok=<counts csv>:<sum>
```

The counts are per bucket (one more than the bounds: the open top bucket). `rss` and `avail` are omitted where `/proc`
cannot be read. One more line at install names the policy: `[PINDIAG] prefix: install policy evict=... supersede=...
telemetry=... ghost_entries=... checkpoint_bytes=...`.

The returning-session memory is bounded (`QWEN_PREFIX_GHOST_ENTRIES`, default 65536 records, 0 for none); the per-request
tables are bounded in code. It costs a few dictionary operations per admission attempt and nothing on the device path.

## Opt-in store policies

Both are off unless set, and off they leave the registry's decisions exactly as they were (`test_qwen_prefix_tiers` pins a
digest of everything observable over a set of conversations against the registry as it was before these existed).

* `QWEN_PREFIX_SUPERSEDE=1` retires, when a newer checkpoint of a conversation lands, that conversation's older ones: the
  candidates are the checkpoints of the same first block that are token-prefixes of the new one, the newest of them (the
  previous turn, which a retry or an edit of the last message resumes from) stays, and so does any that is pinned (this step
  restores it), shared (taken at a gap boundary, so other sessions' prefixes reach it) or a branch point (granted to two
  admissions). The remaining cost is a history rewritten below the previous turn's boundary; `class_ckpt_evicted_superseded`
  counts exactly those misses.
* `QWEN_PREFIX_EVICT=fair` evicts, over the byte budget, the least recently used unpinned checkpoint of the tenant holding the
  most bytes (the tenant is the salt's opaque tag), sparing the checkpoint just stored, instead of the global oldest. The
  default `lru` is the previous behaviour. Evictions by the fair policy count in `evicted_fair`.

A profile may set them in its `env` only beside `QWEN_PREFIX_REUSE=1`, and only to the values above
(`serving_c2_contract.prefix_policy_problems`); any other value stops the engine at start.

## Checkpoints stored in the device's layout (`QWEN_PREFIX_CKPT_PRECONVERTED=1`, default off)

A restore used to put each of the 96 GDN checkpoint tensors (48 layers, state and conv carry) through a single-threaded host
tilize and dtype conversion inside the call, about 1.15 GB/s, and the call is on the critical path of the returning turn.
With the flag on, that conversion happens once, at capture time (off the critical path): the tensor is stored as the
`ttnn` host tensor the device takes (`from_torch` to the device tensor's own dtype and TILE layout, sharded over the mesh as
the restore shards it), and the restore is 96 `copy_host_to_device_tensor` calls and one synchronize. It needs the `h2d`
restore mode (the one the serving model chooses); with the flag on and another mode the model stops at start.

* Size: the stored form is tile-padded (the conv carry's 3 rows become 32), so a checkpoint is charged its stored size,
  about 106.95 MB for the mesh against 78.45 MB unconverted (bf16 state), and the default 8 GiB store holds about 80 of them
  instead of 109.
* Exactness: the stored bytes are what today's conversion produces from the same values. `test_qwen_prefix_model_runtime`
  (PreconvertedCheckpoints) pins that on the CPU fakes; on the card `QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT=1` converts again
  after each restore and compares every tensor with the stored one, uploads, reads the device scratch back and compares it
  with the source, and logs `[PREFIX-AUDIT-CKPT] tensors=96 conversion_equal=1 readback_equal=1` (any 0 stops the engine).
  Each capture stored this way logs `[PINDIAG] prefix: checkpoint preconverted req=<id> pos=<n> tensors=<n> host_bytes=<n>
  convert_ms=<f>`: the conversion moved to the capturing prefill (by that many milliseconds) from the returning turn's restore.
  A conversion that fails leaves the unconverted checkpoint, which a restore converts as before.
* Both names are gate-only until a qualification record lifts it (`serving_c2_contract.prefix_policy_problems` refuses them on
  a traffic profile, and `apply_environment` drops an inherited value under a profile that does not name them).

## The host KV tier (`QWEN_PREFIX_HOST_TIER_GIB`, default 0 = off)

Prefix reuse needs two things for a returning turn: the GDN checkpoint (host RAM, above) and the attention-KV pages of the
prefix, which live in the device pool and are evicted when other sessions need the room. A checkpoint whose pages were
evicted is useless, so the registry dropped it (`evicted_coupled`) and the turn re-prefilled from the start. With the tier on:

1. **Spill.** When vLLM is about to evict a cached KV block whose chain has a checkpoint at or above it (a block with none is
   `tier_spill_dropped_useless`), the scheduler graft remembers it; at the end of that `schedule()` call, with the device idle
   and nothing yet written for the step, the blocks are read from the device in one pass into the tier and the checkpoint at
   the block is kept (`tier_ckpt_kept`) instead of dropped.
2. **Restore.** When a returning turn's trim finds that the best hit lies beyond the device's, the missing blocks are written
   back to freshly taken pool blocks, hashed and cached exactly as if the turn had just computed them, and vLLM's own hit logic
   is asked again: from there on it is an ordinary device hit, under vLLM's own reservation and accounting. There is no
   connector and no external-token path.
3. **Never worse than a miss.** A restore that cannot run (the pool lacks the room, a record failed its digest, the device call
   raised, the kill switch) is a miss, and the turn prefills as it would have without the tier. Three device failures in a row
   latch the tier off for the life of the process and clear it (`tier_latched`).

Tenancy is by construction: a block is keyed by vLLM's chained block hash, which carries the `cache_salt`, so a request under
another salt can never name a tenant's block (the gate checks that no fresh-salt request is ever restored). The tier's byte
budget is the *whole* prefix-state host budget, `QWEN_PREFIX_HOST_TIER_GIB`; the checkpoint store keeps its own
`QWEN_PREFIX_STORE_GIB` share of it and the KV pages get the rest (the GiB must exceed the store's). Inside its share the tier
evicts records no checkpoint reaches first, then by the registry's `QWEN_PREFIX_EVICT` policy (`lru` or `fair`, per tenant); a
record being restored is pinned against that. The pages are held in one anonymous mapping (no `core` dump, grown by use, not
reserved up front), so host RSS rises with what the tier holds, up to its cap; with swap off the cap is the number to size. If the
kernel refuses the one mapping of the whole cap (a strict overcommit policy, a small host), the tier starts anyway and holds the
blocks as separate arrays, still bounded by the cap and a little slower on the first fills; it says so once
(`[PINDIAG] prefix: host tier slab not mapped (...)`).

### Byte-identical restore

* A block's payload is the raw packed bytes of every KV cache tensor for that block (K then V per layer, every chip's page
  range), moved by the `qwen_kv_read` extension's raw block ops (below): no host tensor, no unpack, no repack.
* A digest (sha256) of each record is computed off the critical thread and checked at restore (`QWEN_PREFIX_HOST_TIER_VERIFY`:
  `sample`, one block in sixteen, the default; `all`; `off`). A record that fails is dropped and the restore refused.
* `QWEN_PREFIX_HOST_TIER_AUDIT=1` (gate-only) reads every restored block back from the device and compares it with the bytes
  written, raising and stopping the engine on any difference, and forces `all`.
* The end-to-end proof is the gate's: a returning turn against its cold twin (every output token) and the slot and logits
  digests. The CPU proofs run the real scheduler graft, on a fake pool and on vLLM 0.25.1's own pool and scheduler objects,
  with a device whose block contents are a pure function of the block's hash, and check after every scenario that every
  block the pool serves as cached, and every record of the tier, holds the bytes its hash stands for.

### Flags

| Name | Default | Meaning |
|---|---|---|
| `QWEN_PREFIX_HOST_TIER_GIB` | 0 (off) | the whole prefix state's host budget in GiB; must exceed `QWEN_PREFIX_STORE_GIB`; needs `QWEN_PREFIX_REUSE=1` |
| `QWEN_PREFIX_HOST_TIER_VERIFY` | `sample` | digest check at restore: `sample`, `all`, `off` |
| `QWEN_PREFIX_HOST_TIER_AUDIT` | 0 | gate-only: read back and compare every restored block (forces `all`) |
| `QWEN_PREFIX_HOST_TIER_SPILL_MAX_BLOCKS` | 512 | most blocks one step's evictions may spill; the lowest block indexes of each chain are kept |
| `QWEN_PREFIX_HOST_TIER_MIN_TOKENS` | 8192 | sessions whose checkpoint ends below this are not worth a restore and are not spilled |
| `QWEN_PREFIX_HOST_TIER_MIN_AVAILABLE_GIB` | 16 | no spill while the host's `MemAvailable` is under this |
| `QWEN_PREFIX_HOST_TIER_OFF_PATH` | `kv-tier.off` beside `prefix-reuse.off` | the tier's own kill-switch file: present, the tier latches off and clears, no restart |

`prefix-reuse.off` (the registry's kill switch) also disables the tier: nothing is granted, so nothing is spilled or restored.

### Metrics and log lines

Counters (`qwen_prefix_<name>_total`, timers `qwen_prefix_<name>_seconds_total`): `tier_spill_{flushes,blocks,bytes,ms,known}`,
`tier_spill_dropped_{useless,cap,governor,full}`, `tier_spill_failures`, `tier_restore_{requests,blocks,bytes,ms}`,
`tier_restore_refused_{room,digest}`, `tier_restore_failures`, `tier_digest_{checks,failures}`, `tier_evicted`,
`tier_ckpt_{kept,dropped}`, `tier_audit_{reads,mismatches}`, `tier_latched` (1 once the tier latched itself off). Gauges:
`qwen_prefix_tier_{bytes,entries,cap_bytes,on}`. The existing `qwen_prefix_class_kv_evicted_total` and `qwen_prefix_lost_tokens_kv_evicted_total`
are what the tier is meant to move.

```
[PINDIAG] prefix: host tier on gib=<f> kv_gib=<f> block_bytes=<n> verify=<mode> audit=<0|1> spill_max_blocks=<n> min_tokens=<n> slack=<n> off_path=<path>
[PINDIAG] prefix: host tier IO attached block_bytes=<n> tensors=<n> chips=<n> slice_bytes=<n> fingerprint=<s>
[PINDIAG] prefix: host tier spill blocks=<n> bytes=<n> ms=<f> held=<n>
[PINDIAG] prefix: host tier restore req=<id> blocks=<n> bytes=<n> ms=<f> from=<tokens> to=<tokens>
[PINDIAG] prefix: host tier latched off: <reason>
```

### The device ops, and what they cost

The device side is version 2 of the `qwen_kv_read` extension (`optimisation/ttnn-op/kv_region_read`): `ttnn.qwen_read_blocks_raw`
and `qwen_write_blocks_raw` move named blocks' page ranges between a paged cache tensor and a host buffer (runs of consecutive
block ids are one region transfer each, every chip's shard, one wait; no program runs and nothing is allocated on the device),
and `qwen_block_bytes` says how many bytes one block of one chip is. The image refuses to start with the tier on and the ops
absent. Version 2 was written and tested against fakes, then qualified on the four cards by the first job of
`references/tp4-prefix-tiers-jobs` (Q1, the kvread probe's raw arm): raw bytes read, written into a second cache and read back byte for byte,
the unpacked values equal, no program-cache growth, 13.6 GB/s read and 22.8 GB/s write at 4,096 blocks.

At the production geometry one block (64 tokens) is 32 cache tensors x 4 chips x 17,408 bytes = 2,228,224 bytes, so:

| Session | Blocks | Bytes | Share of a 24 GiB KV budget |
|---|---|---|---|
| 128k (131,072 tokens) | 2,048 | 4.56 GB | 17.7% |
| 254k (253,920 tokens) | 3,968 | 8.84 GB | 34.3% |

Host-side cost, measured on the CPU with the device ops stubbed (`scripts/ci/bench_prefix_tier.py`: the adapter's staging copies,
the store and the digests, at 512 blocks on a busy 20-core host; the device time is NOT in these, and the rates move with the host's
load):

| Step | Rate | 128k | 254k |
|---|---|---|---|
| spill, staging copy into resident slab slots | 9.6-10.7 GB/s | 0.4-0.5 s | 0.8-0.9 s |
| spill, first fill of a slab region (page faults) | 1.05-1.22 GB/s | 3.7-4.3 s | 7.2-8.4 s |
| restore, staging copy from the slab | 4.7-9.4 GB/s | 0.5-1.0 s | 0.9-1.9 s |
| digest at spill (a worker thread, off the step) | 1.5 GB/s | 3.0 s | 5.9 s |
| digest check at restore, `sample` (1 block in 16) | 1.5 GB/s | 0.19 s | 0.37 s |
| digest check at restore, `all` | 1.5 GB/s | 3.0 s | 5.9 s |
| store: put / get / put with eviction | 24 us / 0.5 us / 5-9 us per block | | |

At Q1's device rates (4,096 blocks) one 254k session's spill reads in about 0.65 s and its restore writes in about 0.39 s (a 128k one in
0.34 s and 0.20 s), on top of the host column above and not overlapped with it.

So the cost of a returning 254k turn is dominated by two device transfers of about 8.8 GB (the restore, and the spill of the
blocks it displaces from a full pool, which happens in the same step), the digest check if `all` is on, and then the tail
prefill; against it is the cold prefill of 254k tokens. The first fill of a slab region is page-fault bound, so the first spills
after start are slower than the steady state; nothing is pre-faulted. The gate's `tier-timed` arm measures the whole sequence
at ~32k, ~128k and ~254k (the cold prefill's TTFT, a device-resident hit's, the returning hit's with its restore and displaced
spill) and prints the device rates the host-side table leaves out.

### The gate

`c2_prefix_gate.py --plan tiers` (`tier-attach`, `tier-returning`, `tier-timed`, each also a plan) on
`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-tier-audit`, generated by `make_prefix_tier_profiles.py` from the production
profile (the tier names plus the multi-user SDPA launch's own audit, which the gate's extent audit needs beside W2; gate-only,
no traffic waiver). `tier-returning` builds three ~56k sessions, floods the pool with fresh-salt prompts until every cached block
of the sessions has left the device, brings each session back against its cold twin, and writes the tier's kill switch; it is
NOT_EXERCISED, never a pass, if no returning turn was restored. Two things the gate's oracle (`prefix_judge.Oracle`)
does not know: the tier, and the larger size of a preconverted checkpoint (about 80 fit in the default store, not 109). The arms stay
well under that count, and a returning session that got less than the oracle's Q is explained, not failed, when the tier's or the
checkpoint store's own eviction counters moved. The job pack, its order and its read rules are
`scripts/ci/references/tp4-prefix-tiers-jobs`; `scripts/ci/pin_kvread.py` moves the extension's pinned hash before its image build.

### First card results (the four-card mesh, the production profile plus the tier names)

* Q1, the raw ops: PASS as above. A1, the audited attach smoke: PASS, 0 audit mismatches over 936 publish rounds.
* T1, `tier-attach`: three cold/hit pairs IDENTICAL (solo 3 of 3). Four captures stored preconverted (106,954,752 host bytes each, 60-61 ms to
  convert, at capture); one restore from a preconverted checkpoint (Q=2048 of L=12,427, `restored_ms` 174.5 with the on-card comparison and
  read-back running) with `[PREFIX-AUDIT-CKPT] tensors=96 conversion_equal=1 readback_equal=1` and the program count unchanged. A chain evicts
  nothing, so the tier spilled and restored nothing (`tier_cap_bytes`, `tier_on` only); the arm failed only on the multi-user SDPA audit's
  passing-line rule, which a one-request-at-a-time arm can never satisfy and is now not asked of one (`c2_prefix_gate.sdpa_audit_optional`: only
  while no packed round of two or more users ran; a logged line is still judged).

### Not done

The NVMe tier; the tier on a traffic profile (the contract refuses it until a qualification record exists); the restore through
vLLM's connector API (not needed: the pages are cached before the hit is looked up); a device-side gather that would let the
spill overlap the step; arming anything on the production profile.
