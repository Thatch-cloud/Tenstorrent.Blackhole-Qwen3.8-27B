# Upstream contribution history

PR #2 (`58dce72`) recorded these pointers on 28 August 2026. Review and merge
status here is historical, not a current upstream support guarantee.

| Contribution | Upstream record |
| --- | --- |
| GDN FIR batch-bound correction | [PR 53320 suggestion](https://github.com/tenstorrent/tt-metal/pull/53320#discussion_r3879948665) |
| Reproduced slice failure on the pair | [PR 53319 report](https://github.com/tenstorrent/tt-metal/pull/53319#issuecomment-5451571926) |
| B32 matmul divisibility at TP2 | [Issue 54724](https://github.com/tenstorrent/tt-metal/issues/54724) |
| Short-prompt batched prefill core limit | [Issue 54725](https://github.com/tenstorrent/tt-metal/issues/54725) |

At that snapshot PR 53314 was awaiting re-review; PRs 53319/53320 had no human
reviews recorded. Hardware reproduction is not maintainer endorsement.
The derived patch remains Apache-2.0; repository glue is MIT. See [NOTICE](../NOTICE).
