# tp4-kvread-jobs: the region read's QUALIFY and the P1 control re-run on production bytes

A supplement to `tp4-w1-levern-jobs` (see `ORDER.txt` for the rows, the tags, the fallbacks and the cost). `docs/prefix-audit-cost.md` has the finding (the audit read the whole KV pool per request, 8.4 minutes at 8 x 262k) and the build.

| Job | Question | Answer in |
|---|---|---|
| `Q1-kvread-qualify` | Is `ttnn.qwen_read_blocks` byte-identical to the whole-cache read on the served mesh, compile-free and proportional to the row? | `KV_READ_PROBE verdict=PASS` |
| `P1-CTLR-prod-bytes-...` | Do exactness-shared and lifecycle-evict hold on PRODUCTION bytes (the baselines W-2's cutover rule needs)? | both arms PASS |
| `P1b-CTLR-...`, `P1b-LN-lifecycle-only` | The fallbacks when Q1 did not PASS (no audit). | the arm verdict |

The real-prompt half of the qualification is not a job of its own: the first audited step of every `exactness-shared` arm runs with `QWEN_PREFIX_AUDIT_READ=cross` and is compared byte for byte with the whole-cache read of the same blocks
(`[PREFIX-AUDIT-CROSS] tensors= mismatched=`), one whole-pool read (~8 minutes) inside a job that already boots the engine. A standalone arm exists for ad hoc use (`C2_PREFIX_PLAN=read-qualify`).

Run a job the way W-1's are run: copy the template over `.github/c2-serving-job.env` on a throwaway commit and push the job's tag. No tag is pushed from the branch by its author.
