# tp4/lookup: prompt-lookup drafting for code (QWEN_FAST_LOOKUP_DRAFT)

Branch `tp4/lookup`, from `tp4/next-5`. Default off, byte-identical off. Nothing here has run on a card.

## What it is

Coding output copies its context: a function the user pasted, an identifier defined above, a tool result quoted back. Per user and per
round, a host-side lookup takes the last N tokens of that user's history (prompt plus every committed token), finds their most recent
earlier occurrence and proposes the tokens that followed. When the match is long enough the lookup's tokens replace DFlash2's proposal
rows in the packed verify input. The target verifies every row, so the output is the greedy output either way: the lookup can only change
how many tokens a round commits.

`QWEN_FAST_LOOKUP_DRAFT=<policy>`. The policy is `n<N>m<M>` (`n3m12`): key length N = 2..6, gate M = N..64. The match is the key
extended backwards while the tokens before it agree (cap 64); the lookup is used when that length is at least M. Unset, empty, `0` or
`off` is off; anything malformed is a `ValueError` where the request is built, never a silent off.

A lookup shorter than the ticket (the occurrence sits near the end of the history) is completed with DFlash2's own rows at the same
positions, so the ticket keeps its width: the packed block, its captures and vLLM's scheduler all fix it. The later rows are conditioned
on DFlash2's chain rather than the lookup's, so they only count when the lookup was right to its end.

## How the proposal tokens reach the verify input

The premise "the drafter writes its output into a device buffer the verify reads" is not how this stack works. The drafter's output is
read back to the host, selected there in FP64 (`select_round`, `trace.adopt`), and returned through `DFlashRequestRuntime.__call__` to
`GreedySession.propose`, which builds a `BlockTicket`. The verify input is then staged from that ticket, each round, on the host:

1. `FastRequest.prepare` (`serving_fast_request.py`) calls `session.propose`; under the flag it then calls
   `RequestLookup.apply(session, ticket)`, which returns the same ticket or a replacement built by `prompt_lookup.replace_proposals`
   (same position and seed, a new epoch, `session.pending` set to it: the twin of `GreedySession.narrow`, because the harness is pinned).
2. `FastRunnerBridge.drafts` hands vLLM `ticket.tokens[1:]` from that ticket and `admit_scheduler_output` compares the scheduler's tokens
   with it, so vLLM, the admission check and the verify all see the lookup's rows.
3. At verify time `PackedVerifierEngine.segment_users` reads `ticket.tokens` per entry and `stage_packed` / `write_packed` copy them into
   the captured token input (`fixture.tokens`, a uint32 buffer allocated before the first capture) with `copy_host_to_device_tensor`.
   QWEN_FAST_PRESTAGE only pre-writes placeholders for the tokens; the verify-time diff writes them.
4. `GreedySession.commit` judges `ticket.tokens[1:]` against the target's predictions, so acceptance is measured on the rows that ran.

So the substitution is a host edit of the ticket before anything reads it, and there is no device allocation at all: no buffer exists
that a later capture could land on. The drafter still runs every round and its history stays current, because `DFlashRequestRuntime.publish`
advances it from the target's verified features at the committed prefix, never from the proposal tokens.

## Cost

The index is built once per request, synchronously, when the request is created: `serving_request_factory` calls `prompt_lookup.for_request` after the
prefill, on the engine step thread. It is NOT in the shadow of the prefill: while it builds, every decoding seat waits. A dict from the packed integer key
(18 bits a token) to the end index of the most recent earlier occurrence, plus the history list. Measured on a laptop CPU with 123,000 random tokens (the
worst case: every key distinct): 0.27 s to build, 14 MB; a 253,920-token prompt would cost about 0.6 s and 30 MB, about two eight-seat rounds of stall per
admission. So the prompt is capped: only the last `MAX_INDEXED_PROMPT` = 32,768 prompt tokens are indexed (about 0.07 s and 4 MB at the cap), and every
committed token is indexed whole. The offline estimate's prompt-inclusive history is the uncapped one; at 4k and 32k prompts (the timed arms) the cap changes nothing. A round costs about 5 microseconds to propose (a dict probe, a backward comparison of at most 64 tokens, a
copy of 15) and about 36 microseconds with the six committed tokens appended to the index.

## Flag off

`serving_request_factory` imports nothing and builds nothing; `FastRequest.lookup` stays `None` and `prepare` takes one `is not None` test.
`test_prompt_lookup` runs the same requests with the flag unset, `off` and `0` and requires identical events, outputs, epochs and no lookup
log line.

## Log lines and the smoke rule

* `[LOOKUP-DRAFT] engaged policy=n3m12 request=<id> history=<tokens>` once per request.
* `[LOOKUP-ROUND] request=<id> position=<p> source=lookup|dflash2 match=<m> offered=<k> proposed=<k> committed=<c>` per user per round, logged when the
  round's commit is known (at the next round, and at close for the last), so a timed window measures the real tau. A ticket discarded before its
  verify is not a round.
* `c2_smoke_check.lookup_problems`: flag off means no lookup line; on means an engaged line naming the policy and well-formed round lines.
* `scripts/ci/lookup_tau_report.py A B`: tau per arm from the `[PACKED]` emitted counts (live 4, each request's last round dropped) and the B arm's
  rounds split by source.

## Profile and window

`c2-packed-tp4-best-lookup` is `c2-packed-tp4-best-strace` plus `QWEN_FAST_LOOKUP_DRAFT=n3m12`, nothing else (gate only; a test holds the
difference to that one key). The job pack `scripts/ci/references/tp4-lookup-jobs`: X0 reads the rig status (four boards, no holder); B0 builds `tp4-lookup-1`; L1 is the
audited smoke (rescan before reset); TL1..TL4 are the paired ABAB of best-strace against best-lookup on coding prompts at 4k and 32k (exactness read from
the SMOKE_JSON hashes, tau from `lookup_tau_report.py`); Z resets. No job stops or starts the node agent, hands back or deploys. `prompt_lookup.py` is in `docker/qwen-c2-overlay.txt`.

## The offline estimate (`optimisation/lookup/`)

`build_tapes.py` turns a tau-lab run (turns, outputs, container log, optionally the prompt ids) into a tape file; `lookup_sim.py` replays the logged
DFlash2 rounds against the lookup (the served `prompt_lookup.TokenLookup`, imported not copied), for key lengths 2..6 and gates 2..32, in two modes:
`recorded` (every policy at the logged round starts: simple, and it overstates a lookup that commits a long run) and `walk` (token-conserving, with the
restart model the earlier alt-drafting analysis fitted). The `dflash` row of the walk reproduces the lab's published figures exactly (pooled tau 4.485,
33,360 counted rounds, per-turn p10 3.474), which is the check that the harness replays the log correctly.

Result on the 336 thinking-on turns of the lab's A1/A2 arms, 3 restart draws, spliced tickets:

* With committed tokens only (no prompt ids at hand when this was written), the lookup finds nothing worth having (not a strict lower bound: extra matches from the prompt can pass the gate and still lose to DFlash2): gate 16 gives +0.002 to +0.003 tau
  (+0.1%) for every key length; gates below 8 lose (n3 gate 3: -3.4%, n2 gate 2: -10.7%).
* The gain is in the prompt. The same lab data with the prompt ids, in the earlier alt-drafting analysis (longest match instead of most recent occurrence, no
  completion with DFlash2 rows), gives lookup-first hybrids of +0.133 to +0.135 tau (+3.0%) at gates 8-16, used on 6% of rounds at gate 16, and almost all of it
  from sources more than 2,048 tokens back (beyond the drafter's window). Lookup alone is 1.96 against the drafter's 4.49. That +0.13 comes from the earlier
  longest-match matcher, not the served n3m12 most-recent policy, so n3m12 is not validated by it: TL1-TL4 are what measure it.
* The recorded-start count says +0.44 for the same gate; the walk says +0.13. Trust the walk.

Re-run with prompts (private inputs, stdlib only, any python 3.7+):

    python optimisation/lookup/build_tapes.py --turns turns.jsonl --outputs outputs.jsonl --log server.log --ids a1.ids.jsonl.gz a2.ids.jsonl.gz --out tapes.jsonl.gz
    python optimisation/lookup/lookup_sim.py tapes.jsonl.gz --mode walk --seeds 4 --detail

## What this does not settle

The hybrid's gain is about +3% tau at a cost of well under a millisecond, which is worth having but is nowhere near the 8 x 75 bar, which needs a
drafter that misses half as often. The two-candidate verify (the drafter's block plus a lookup branch, +8% in the same simulation) needs kernel work and is
not part of this branch.
