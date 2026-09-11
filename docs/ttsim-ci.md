# TT-Sim on CI

Simulator work runs on the existing AMD runner, not the developer PC.

| Setting | Value |
| --- | --- |
| Runner | `thatch-build-amd64-02-cp-temp` |
| Container CPU limit | 16 CPU equivalents; not exclusively pinned cores |
| Container RAM limit | 64 GiB |
| Device access | None; no Tenstorrent devices mounted |
| Container network | Disabled; pinned simulator assets fetched before launch |
| Scheduling | Serialized with two-card hardware experiments |
| Evidence | Logs, JSON reports, asset hashes and exit status uploaded by CI |

The dedicated `qwen-ttsim.yml` workflow is on the experiment branch. Until it
is registered on the default branch, dispatch through `qwen-experiments.yml`:

```powershell
gh workflow run qwen-experiments.yml --ref <approved-immutable-tag> `
  -f suite=learned-attention -f simulator_only=true -f simulator_fusion_t16=true
```

Add `-f simulator_target_math=true` for the explicit target-native math-mode
comparison. Each immutable workflow tag must be added to the restricted runner
group without removing existing grants. Neither route enables serving changes
or provides hardware throughput measurements.

The initial CI run is [34558325869](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34558325869).
Its outcome must be checked before qualification; dispatch is not a passing test.

The preceding local run was deliberately stopped to free the developer PC.
It completed packed-weight checks and T1/T8/T16 changing-input replay, but not
the complete T32 gate or clean successful exit. Its partial evidence is not a
full simulator pass. The temporary local packer modification was restored.
