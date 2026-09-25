# The card-B runner

Card B (board `blackhole-F36F768B9A5CAFA0`, PCI `0000:f4:00.0`) is the rig's PCIe-only qualification card. It
has its own GitHub runner so card-B tests queue through CI instead of being run by hand over SSH, and run in
parallel with the serving-pair gates on cards M and A.

| | |
| --- | --- |
| Runner | `thatch-qwen-cardb-01`, systemd service on `thatch-control-plane-prod`, install dir `/opt/thatch-actions-runner-cardb` |
| Labels | `self-hosted`, `Linux`, `X64`, `thatch-qwen-p150a-card-b` |
| Runner group | `qwen-card-b` (org): this repository only, workflows restricted to `qwen-card-b.yml` tags |
| Workflow | `.github/workflows/qwen-card-b.yml`, triggered by tags `experiment/card-b-v*` |

## Queueing a test

1. Edit `.github/card-b-job.env`: the harness (`CARD_B_HARNESS`, a card-B runner script that uses the
   `qual_card` board selection, such as `optimisation/ttnn-op/<dir>/run_card_b.sh`), its arguments
   (`CARD_B_ARGS`) and any extra environment (`CARD_B_ENV`).
2. Commit, then push the next tag: `git tag experiment/card-b-vN && git push origin experiment/card-b-vN`.
3. Repeat for more tests. There is no concurrency group: the one runner takes one job at a time and GitHub keeps
   the others queued in order, cancelling none.

Results come back as the run's `card-b-<run id>` artifact (the harness's `RESULTS` directory plus
`harness-console.log`).

## What the workflow enforces

- It runs only on `thatch-qwen-cardb-01` and only harnesses under `optimisation/ttnn-op/<dir>/` or `scripts/ci/`
  that use the `qual_card` selection.
- `QUAL_CARD` is always card B; the job file may not set `QUAL_CARD`, `ALLOW_SERVING_CARD` or `RESULTS`. The
  harnesses refuse cards M and A without `ALLOW_SERVING_CARD`.
- Card B is resolved by board id and checked against its PCI address, and nothing may hold it at the start.
- Results are written under `/tmp/card-b-results/<run id>` (never the workspace: a container's root-owned files
  there break the next checkout), copied into the artifact, then deleted.
- Every container the job started is removed at the end, whatever happened.
- Nothing resets any card. A wedged card B needs a person (see the harness's printed reset hint).

## Allowlist

The group's `selected_workflows` lists each allowed tag, as runner group 3 does for the pair: tags
`experiment/card-b-v1` to `v200` were added when the runner was created. Add the next range before reaching v200.

## Undo

On the rig: `cd /opt/thatch-actions-runner-cardb && sudo ./svc.sh stop && sudo ./svc.sh uninstall && ./config.sh remove`,
then delete runner group `qwen-card-b` in the organisation settings.
