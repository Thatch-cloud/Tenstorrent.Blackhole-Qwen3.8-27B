# Three runs, one hole: a host flag that never crossed into the container

2026-09-22. Written after run 35679222511 (Lever N v36) spent a rig slot testing
nothing, for the third time by the same mechanism.

## The finding

The v36 arm was the first run that could have exercised M1's resumable prefill.
It grafted all seven files correctly. It still prefilled every 32,768-token
prompt whole. Its own artifact says why - the server argv, recorded in
`m3native-gate.json`:

```
--max-num-batched-tokens 33024
--no-enable-chunked-prefill
```

Chunked prefill was explicitly disabled on the run whose entire purpose was to
test chunked prefill.

## The mechanism

`lever_n_m3native_run_arm.sh` reads `M3NATIVE_PREFILL_CHUNK_TOKENS` **on the
runner host** and uses it to decide what to mount:

```bash
if [ -n "${M3NATIVE_PREFILL_CHUNK_TOKENS:-}" ]; then
  lever_n_mounts+=(--mount .../model.py ...)          # the graft: correct
  lever_n_mounts+=(-e TT_M1_FORCE_CHUNKED_PREFILL=1)  # the platform opt-in: correct
fi                                                    # the variable itself: never passed
```

`lever_n_m3native_gate.py` is the thing that turns that variable into
`--enable-chunked-prefill`, and it reads it with `os.environ.get` **inside** the
container. It saw `None`, took the branch documented as "byte-identical without
it", and launched a stock server. `start_pos` stayed 0, so the graft's own
dispatch logged `[M1] prefill path: one-shot prefill_paged_slots` and the
resumable path never executed.

Two more flags were dead the same way: `M3NATIVE_TTFT_MAX_S` and
`M3NATIVE_TTFT_MAX_STALL_S`, so neither TTFT ceiling could ever have been
asserted.

## Why this is a verdict and not a bug report

It is the third occurrence, and the first two were already written down in this
repo's own comments without the pattern being named:

| Runs | Symptom | Same cause |
|---|---|---|
| 35415521079, 35415811328 | never chunked a 5,918-token prompt "despite being asked to" | policy override, flag not effective |
| 35423537994 | prefill chunk 2048 vs 4096 measured 32.64 s vs 32.61 s | patched constant was dead; both arms ran the same fallback |
| 35679222511 | `one-shot prefill_paged_slots`, TTFT ceilings unassertable | host env never reached the container |

Every one of them produced a plausible-looking result. The 2048-vs-4096
measurement is the worst of the three: it did not fail, it returned a *number*,
and that number said "chunk size does not matter" when the truth was "the arms
were identical".

A run that fails is cheap. A run that succeeds at the wrong thing costs the
verdict too.

## The guard

`scripts/ci/test_m3native_arm_env.py`, allowlisted in `qwen-integration-cpu.yml`:

1. Derive the container-side script list from the arm's own `dst=/bench/*.py`
   mount flags, so it cannot drift from what is actually mounted.
2. Walk each script's AST for `M3NATIVE_*` environment reads.
3. Fail if any of them is not passed through as a docker `-e NAME=`.

Two details are load-bearing. It resolves reads **through helper functions**,
because `m3native_ttft_profile` reaches its ceilings via
`_threshold(environ, FLAG_MAX_STALL)` - a literal-only scan finds
`M3NATIVE_PREFILL_CHUNK_TOKENS` and misses the other two. And it chases the
call rather than matching every `M3NATIVE_`-prefixed constant, because the
gate's `M3NATIVE_GATE_JSON_BEGIN` stdout markers are printed, never read, and
demanding a passthrough for a sentinel would be noise that gets suppressed.

Negative control, with only the three passthroughs reverted:

```
read inside the container but never passed to it:
  M3NATIVE_PREFILL_CHUNK_TOKENS (lever_n_m3native_gate.py);
  M3NATIVE_TTFT_MAX_S (m3native_ttft_profile.py);
  M3NATIVE_TTFT_MAX_STALL_S (m3native_ttft_profile.py)
```

4 of 5 tests fail and each flag is named with the file that reads it.

## The reporting error, separately

I stated that the v36 arm "produces `--enable-chunked-prefill
--long-prefill-token-threshold 2048 --max-num-batched-tokens 2048`". The gate
does build exactly that - from a variable that never arrived. I read the argv
the code *intends* instead of the argv in the artifact, which had been sitting
in `m3native-gate.json` the whole time.

The rule that follows: when an arm claims to change engine behaviour, read the
recorded argv before reading any measurement. It is one `jq` away and it is the
only thing that distinguishes the arm under test from a stock run.

## What v36 did settle

Image drift, which was recorded as unverifiable from this repo. M1's anchors
were developed against image `bd878710e15c`; the m3native lane runs
`e41ef884f4c8`. All seven files grafted with every `replace_once` anchor
matching exactly once and every result `ast.parse`-clean. That question is
closed.

## Open, and deliberately not chased first

The 60-second timeout at the second user's admission (issue 48, now 3 of 9
runs). It is intermittent, so it is a retry tax, not a wall; the dead flag was
100% and cost nothing to fix. One caveat on attribution: v36 had the grafted
files mounted even though the chunked branch was never taken, so it does not
fully clear Lever N of causing the hang. Run 35681324335 (v37) is the
disambiguator.
