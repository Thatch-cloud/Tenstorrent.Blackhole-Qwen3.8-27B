# Agent work profiling: capture real coding-agent sessions, replay them on the cards

Status: design (2026-10-11). Nothing built yet. Owner direction: design it, track it, run capture on a dedicated capture box when it is
available; replay runs inside the existing card-job harness.

## 1. Why

Every decode and prefill number so far comes from synthetic shapes: eight users on about 3.5k-token coding prompts
(`concurrent8_steady`), eight users at 32k and 128k (`concurrent8_code_32k`, `concurrent8_code_128k`), solo prefill ladders and
prefix-gate plans. Real agent traffic is different in ways those shapes do not show: many short turns, tool gaps from seconds to
minutes, contexts that grow turn by turn to 100k+, sessions that go idle and come back, and history that may not re-render to the
tokens the model actually processed (reasoning stripped by the chat template, tool calls re-serialised). The questions this answers:

| Question | Who needs it |
|---|---|
| Per turn: prompt tokens, reused (cached) tokens, the re-prefilled tail, TTFT, decode tok/s, at real context lengths | every lever decision; exact-resume scoping (`docs/tp4-exact-resume.md`) |
| Where the re-sent history first diverges from what the model processed (template, tool-call rendering) | exact resume, pinned prefixes |
| The real tool-gap distribution | the adaptive session idle hold (platform side) |
| How often a session is evicted from the card pool, and what its return costs | the host-RAM / NVMe tiers and per-tenant quotas |
| End-to-end agent time per task (model time vs tool time) | what users feel; the 8 x 75 tok/s target in context |
| Task pass rate per stack version | quality regressions that byte-exactness gates cannot see (template, sampler, drafter settings, long-context handling) |

## 2. Shape: capture rarely, replay every window

Two layers, deliberately separate:

1. **Capture** (a real agent solving real tasks, on a capture box, against an OpenAI-compatible endpoint serving Qwen3.8-27B):
   produces a pass rate and a corpus of real multi-turn sessions with their timing. Slow, non-deterministic, needs task containers.
2. **Replay** (deterministic, no task environment): re-sends the recorded request sequences with the recorded tool gaps, N sessions
   at a time, against any image, as a smoke test of the card-job harness. Reproducible, comparable ABAB, cheap.

A custom thin harness, not an agent framework: frameworks (OpenHands, SWE-agent, Aider) are built to score agents, are heavy and are
hard to replay; profiling needs realistic capture and exact replay.

## 3. Capture

* **Agent:** a minimal agent loop (one bash tool, a file-edit tool, a submit tool; the mini-SWE-agent pattern) speaking the OpenAI
  chat-completions API with tools, streaming on. Two configurations: thinking on and thinking off. Greedy (temperature 0) for the
  corpus; the eval also records a sampled configuration if wanted.
* **Task set:** start with a fixed 50-task subset of a public benchmark (SWE-bench Verified, Python) so traces are public-safe and
  comparable; later an internal set from our own repositories (traces then stay private, never in this repository). No customer
  traffic without consent.
* **Concurrency:** 8 agents (the seat count), and 12 for an eviction/refusal run.
* **Task containers:** one container per task (repository at the base commit, the task's test command), CPU and disk heavy: the
  capture box only, never a serving host or the card rig.
* **Recording proxy:** an OpenAI-compatible pass-through between the agents and the endpoint. Per request it records: session id,
  turn index, wall-clock send time, the full request body (messages, tools, sampling), the streamed response with per-chunk receive
  times (first token, inter-token), usage (prompt, completion, cached prompt tokens), the response id, and the tool gap (previous
  response end to this request start). One JSONL file per session; a manifest per capture run (endpoint, model, image/profile id,
  agent config, task ids, pass/fail per task).
* **Server join:** the response id is joined with the serving engine's own lines (prefix matched length, Lever N splits, tier
  restores, seat refusals) where the capture endpoint is one of ours.

## 4. Corpus format (versioned)

`agent-trace/v1`: per session `{session, task, config, turns: [...]}`, per turn `{t_send_s, messages, tools, sampling, max_tokens,
recorded: {prompt_tokens, cached_tokens, completion_tokens, ttft_s, tpot_s, finish}, tool_gap_s}`. Messages are stored whole (not
token ids) so the corpus survives tokenizer and template changes; token counts are recorded for reference. A checker validates
schema, monotonic times and that each turn's messages extend the previous turn's (reporting where they do not: that is the
divergence the exact-resume work needs).

## 5. Replay (the `agent_replay` smoke test)

* **Determinism:** each turn sends the RECORDED history (recorded assistant messages, recorded tool results), never the replay's own
  output, so every prompt is identical across arms. The replay's generated output is capped at the recorded completion length
  (`max_tokens`, `ignore_eos` so decode length matches) and hashed per turn for the exactness comparison against the control arm.
* **Timing:** after each response, wait the recorded tool gap (optionally scaled, e.g. x0.5 to compress a run); sessions start on the
  recorded stagger or a fixed offset. N sessions concurrent (8 default; more to exercise eviction and the seat cap).
* **Per-turn output:** prompt tokens, cached tokens (`usage.prompt_tokens_details.cached_tokens`), server matched length (from the
  engine line), TTFT, decode tok/s, 429s and `Retry-After`, plus the round time and tier/eviction counters from the engine log.
* **Aggregates:** TTFT p50/p90 by context bucket (<32k, 32-128k, >128k), per-session decode tok/s, reuse ratio, evictions and their
  return cost, refusals, and per-session end-to-end time (sum of TTFT + decode + recorded tool gaps).
* **Judge:** no stall over the Lever N gap floor (8 s), no request error, no refusal unless the run asks for over-subscription; the
  report is a measurement, not a gate. `agent_replay_compare.py A B [C D]` reads an ABAB pair: per-bucket TTFT ratio, decode ratio,
  per-turn output-hash equality, the control-to-control floor.
* **Corpus delivery to the card job:** public-task corpora as a fixture (compressed, size-capped) in the repository; private corpora
  mounted read-only on the runner from a path the job names by key (never committed).

## 6. Quality eval

The capture runner doubles as the eval: pass rate per task set per stack version (image + profile), run at each ship candidate and
weekly. A drop with byte-exact gates green points at template, sampling, context handling or drafter configuration.

## 7. Phases

| Phase | What | Where | Effort (estimate) |
|---|---|---|---|
| P0 | Trace schema + checker, recording proxy, replayer against a fake OpenAI server, CPU tests and CI | repository, CPU only | 2-3 days |
| P1 | `agent_replay` smoke test in the card harness, compare tool, a seed corpus (a few public tasks captured against any Qwen3.8-27B endpoint), first card run | cards, one window slot (~45 min per arm) | 3-4 days |
| P2 | Capture runner on the capture box: task containers, 50-task subset, thinking on/off, 8 and 12 agents; eval report | capture box | 3-5 days once the box is up |
| P3 | Standard window pack: every lever ABAB also runs `agent_replay`; tool-gap histogram feeds the adaptive hold; divergence report feeds exact resume | cards + reports | ongoing |

## 8. Decisions

1. Capture endpoint: any Qwen3.8-27B OpenAI-compatible endpoint works for capture (the corpus stores messages); one with our engine
   lines gives the server join. Capture load is real traffic: schedule it, use an internal tenant key, never a customer's.
2. Task set: public subset first; internal repositories later with private storage.
3. Trace retention and where private corpora live.
