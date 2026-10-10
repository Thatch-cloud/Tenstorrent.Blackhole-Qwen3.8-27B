# Prefix checkpoint store: telemetry and hygiene

The GatedDeltaNet (GDN) checkpoints that make a prefix hit exact live in `qwen_prefix_registry.PrefixRegistry`, one per
engine process. This note covers what that store now measures about its misses, the two opt-in policies that keep it
useful, and the size arithmetic behind its budget. Nothing here changes what a hit serves: a checkpoint is still
token-verified before it is granted, and a retired or evicted checkpoint is only ever a miss.

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
