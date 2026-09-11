# T32 simulator runtime admission

The experimental 31-query Markov gate uses the immutable CPU CI image
`sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
It does not inherit the old local Markov probe's runtime qualification.

Run 34589120590 failed before simulation because Python 3.10 lacks
`hashlib.file_digest`; streaming SHA-256 replaces that call.
Run 34589267015 then rejected the old binary pin before opening the mesh.
Audit run 34589455129 reports these exact image contents:

| File | SHA256 |
| --- | --- |
| `build_Release/lib/_ttnncpp.so` | `f65ac9e332d34ff462a051a021221fc12377b05711dc67d1faa5aa6fe37858c3` |
| `build_Release/ttnn/_ttnncpp.so` | `d6c53113a104719a442b4d4a9ec2b344cdd0e00daa1e4d907afb9c13d1e531d9` |
| Original packer | `87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181` |

The new probe pins both paths separately and retains before/after hashes of
embedding, matmul and argmax sources. No binary is copied over another and no
hash check is disabled. These pins establish reproducibility, not numerical
correctness: the new eager oracle and changed-input replay gates still must pass.
The initial gate uses 64 synthetic vocabulary entries. Full-vocabulary learned
weights, wider attention, every-prefix target state and combined PP/CTX/TG
remain unqualified. No serving defaults change.

## Completed component gates

| Run | Gate | Independently checked result |
| --- | --- | --- |
| 34589904935 | 31-query Markov, synthetic vocabulary 64 | 186 eager and 248 replay query/chip comparisons; input/weight/stale controls pass |
| 34590151457 | T32 folded target attention at CTX4096 | 8 replay, 24 distinct mask-bundle and 4 KV-integrity checks pass against native B1 |

Both exit cleanly with zero status; recorded probe sources match their immutable
CI revisions (`cab5126` and `8e90139`). Markov report SHA256:
`ce24fcf9924258daa703f74e08cf6570f1e5d872102dc5a952eebd4f05831065`.
Attention report SHA256:
`774a53bf54fcbb7fe5c02fb1f358be63c9fcc0254679f19543232d4079429b1b`.

The learned full-vocabulary Markov gate is queued as 34590621276, using the
cached checkpoint read-only. Dispatch now uses `simulator_t32` with values
`none`, `t32-markov`, `t32-markov-learned` or `t32-attention`, avoiding GitHub's
25-input limit. These component passes do not qualify complete T32 drafting,
every-prefix target state, coding quality or combined PP/CTX/TG.
