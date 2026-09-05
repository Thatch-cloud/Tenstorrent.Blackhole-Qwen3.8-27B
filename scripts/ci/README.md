# Qwen hardware CI inventory

The manual workflow targets the existing organization runner
`thatch-build-amd64-02-cp-temp` using its dedicated custom label
`thatch-qwen-p150a-pair`, plus an exact-name check before checkout. Do not put the
label on shared ARC runners. It was added through the GitHub API without removing
existing labels or restarting the runner. Re-registration must preserve this label.

Access gate discovered on 2026-09-05: this repository is public, but the runner's
Default organization group has `allows_public_repositories=false`. The first manual
run stayed queued and was cancelled before any host steps executed. The dedicated
label does not override repository access policy. Before hardware execution, use an
approved private-repository workflow or obtain explicit approval for narrowly scoped
runner access; do not enable public repositories across the shared Default group.

Thatch.Server owns runner provisioning and isolation. Its runner template enables
`PrivateDevices`, `ProtectHome` and an isolated HOME. This workflow does not change
those settings or assume operator weights/caches reside in the runner's HOME.

The script inventories Docker containers and their mount paths without printing
environment variables. It selects an already-local Qwen/TT serving image by its
immutable image ID; it does not pull an image. A disposable, network-disabled,
read-only, capability-dropped container with a numeric unprivileged UID reads
daemon-host `/dev/tenstorrent` and `/sys` metadata through read-only mounts.
The image entrypoint is overridden. No accelerator is opened or reset.

Only the probe container created by this invocation is removed on exit. Production
containers are never started, stopped or modified. A missing local image, missing
host devices, inaccessible Docker daemon or failed probe causes a failing job.
Inventory is not a test of device idleness, P150A harvesting, fabric health or model
performance. Container device mappings are incomplete evidence of device users;
an exclusive-use check is still required before accelerator execution.

GitHub concurrency serializes this workflow only, not jobs in other repositories
or host services. The dedicated runner label establishes placement, not exclusive
ownership of the cards. Keep serving disabled until the operator restores it.

Host-only script tests: `python3 scripts/ci/test_qwen_inventory.py`.

## Hardware correctness

The manual `hardware` suite opens both allocated cards using a pinned local image.
It audits the runtime, checks fused operators on each chip, validates chunked
prefill recurrent/KV state, and checks eager/traced all-gather over the QSFP-DD
fabric. The asymmetric PCIe x16/x4 attachment is not the inter-card transport.
This is not a full-model decode throughput benchmark.

By default, a read-only host-process descriptor scan must prove no device owners.
If the operator has explicitly allocated both cards, set `cards_allocated=true`.
That mode verifies the physical board mapping but deliberately does not inspect
host processes or claim an OS-enforced reservation. It requires no host PID
namespace or ptrace capability. Do not infer idleness from TT-SMI activity or an
unreadable process scan. Neither mode resets cards or changes serving services.

Hardware entry-point guard tests require Python 3.10 or newer:
`python3 scripts/ci/test_hardware_guards.py`.
