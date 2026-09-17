# Same-recipe context ladder

The ladder measures the complete single-stream generation pipeline, not isolated kernels.

| Setting | Value |
|---|---|
| Hardware | Two P150A cards, four fabric links |
| Prompt contexts | 4,096; 8,192; 16,384; 32,768; 65,536; 131,072; 262,144 |
| Generation headroom | 256 tokens |
| Runtime | Shared Q/K, norm prefetch, incremental publication, tail-first draft assembly, scalar reciprocal, T16 verifier and compact target scratch |
| Per context | One fresh full-model feature/state audit, then two complete timed requests |
| Changed runtime setting | `QWEN_DSPARK_REQUEST_CONTEXT` only |
| Report | PP (prefill tokens/s), CTX (actual prompt tokens), TG (committed generation tokens/s) |

All rows stage identical source files. The pinned corpus prevents harness edits from changing the workload. Longer contexts repeat that corpus where necessary, preserving the final coding task. This is a synthetic latency workload, not held-out coding-quality certification, and its numbers are not directly interchangeable with older live-source prompts.

The retained 32K component reports verify unchanged kernel sources; they do **not** establish correctness at another geometry. Every row must pass its own full-request audits, exact emitted tokens, state checks and clean shutdown. Positional limits and capacity failures remain failures, never implicit truncation or RoPE extension.

The matrix runs one hardware job at a time and continues after a failed row. Matrix execution order is not guaranteed. Explicit benchmark timers are removed; GitHub's platform job limit and setup/cleanup safeguards still apply. Serving defaults are unchanged.

Workflow: `.github/workflows/qwen-combined-ladder.yml`, triggered by `experiment/combined-ladder-v1`.
