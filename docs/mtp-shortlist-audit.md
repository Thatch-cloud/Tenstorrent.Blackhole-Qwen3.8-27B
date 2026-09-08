# MTP short-block follow-up

## Source audit

NInfer branch `feat/qwen3.8-nvfp4full`, pinned at
`53d4efcfb4a4eb84e0a3938fdae910e7f7557e44`:

- [Draft head conversion](https://github.com/cometkim/ninfer/blob/53d4efcfb4a4eb84e0a3938fdae910e7f7557e44/tools/convert/qwen3_6/common/draft_head.py)
  selects head rows using corpus token frequencies and carries an explicit global-ID map.
- [MTP execution](https://github.com/cometkim/ninfer/blob/53d4efcfb4a4eb84e0a3938fdae910e7f7557e44/src/targets/qwen3_6/impl/runtime/mtp_impl.h)
  retains draft/target hidden buffers and performs autoregressive proposal steps.

Our `speculative-decoding` branch already contains `Qwen36MTP`, paged draft KV,
hidden retention, and traced-draft groundwork. Its generation loop uses the full
target LM head and gathers the full vocabulary for each draft. Reuse that work;
do not start a second MTP implementation or confuse it with lookup drafting.

## Bounded next experiment

| Item | Requirement |
| --- | --- |
| Draft lengths | K1 and K3 against corresponding exact target verification widths |
| Draft head | Full vocabulary control versus a fixed precomputed shortlist |
| Ranking | Separate training/calibration data; never future target answers from the evaluated request |
| Special tokens | Required IDs cannot be silently discarded to meet the shortlist size |
| Target | Full vocabulary, unchanged weights and exact greedy verification |
| State | Correct MTP catch-up and rollback at every rejection position |
| Metrics | Draft/head, verify, commit, acceptance and total committed TG; include zero-acceptance rounds |

`draft_vocabulary.py` implements host-only shortlist selection, weight-row packing
and global-ID mapping. Rows are stored in ascending token-ID order for deterministic
ties. It is not yet a device draft head, integrated MTP runtime, or speed result.
Reducing draft vocabulary can reduce acceptance; measure the complete trade-off.
