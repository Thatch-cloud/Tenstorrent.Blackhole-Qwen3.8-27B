# tp4-p2p-jobs: the peer-to-peer KV restore's card job

| Job | Question | Answer in |
|---|---|---|
| `P3a-p2p-restore` | Does an NVMe object read O_DIRECT into a card's staging buffer through the published BAR4 window arrive byte for byte, check against the host's checksum, copy into the KV caches, and is a flipped byte on the drive refused with nothing written? | `P3A verdict=PASS` |

`docs/tp4-p2p-host.md` is the design and section 8 is this job. The job key is `C2_FABRIC_PROBE=p2p-restore` (with `C2_CARDS=quad`, `C2_ACTIONS=fabric`; add `reset` before it after a failed run, as for the other
fabric probes). The drive and the cards are the runner keys `C2_NVME_DEVICE=status` and `C2_P2PMEM=auto` (`scripts/ci/c2_cache_drive.py`; the words keep every by-id name, serial and PCI address out of the pushed job file); the template has them.

Run it the way the other fabric probes are run: copy the template over `.github/c2-serving-job.env` on a throwaway commit and push the job's tag. No tag is pushed from the branch by its author.
