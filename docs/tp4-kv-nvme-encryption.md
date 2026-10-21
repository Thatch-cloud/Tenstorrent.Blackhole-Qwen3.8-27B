# tp4/kv-nvme-crypt: encryption at rest for the NVMe KV tier (scoping)

Status: **design and measurement only.** Branched from `tp4/kv-nvme-1` (`44a6ddbb`). No line of the tier changes on this branch; every code change below is a plan. The tier itself is described in `docs/tp4-kv-nvme.md` (cited as "the tier note", section numbers there).

Labels: **[M]** measured for this note; **[V]** verified by reading source or vendor text; **[D]** derived by arithmetic from [M]/[V]; **ESTIMATE**; **ASSUMPTION** (a card window or the drive itself must confirm).

## 0. Summary

| | A. Opal on the cache drive | B. Software per-tenant keys | C. Hybrid A + B | D1. NVMe crypto erase only | D2. B, decrypted on the card |
| --- | --- | --- | --- | --- | --- |
| Stolen drive (powered off, alone) | **yes** | **yes** (contents; the index metadata stays readable) | **yes** | no | yes |
| Decommissioned / RMA drive | **yes**, also a dead drive (it stays locked) | **yes**, also a dead drive | **yes** | only while it still accepts commands | yes |
| Tenant deletion request | logical delete + TRIM only | **crypto-shred per tenant** | per tenant for sealed objects | no | per tenant |
| Compromised host root (while running) | no | no (limits it to keys resident at the time; shredded tenants stay shredded) | as B | no | as B |
| Operator | physical only (= stolen drive) | physical only; root = root | as B | no | as B |
| Shred granularity | whole drive (one range; section 1.3) | tenant (and DEK generation) | tenant for sealed objects, drive for the rest | namespace = whole drive | tenant |
| Host CPU, 254k restore (8.56 GB) | 0 | ~1.05 big-core-seconds (AES-256-GCM, [D]) | as B | 0 | ~0 (card) |
| Host CPU, write-through (~420-460 GB/day) | 0 | ~0.07% of one core on average ([D]); ~0.7 core at the drive's peak rate | as B | 0 | ~4x B (ChaCha20) |
| P2P restore (NVMe to card) | **unchanged** | **impossible** for sealed objects | P2P only for objects left unsealed | unchanged | possible, needs a new device kernel |
| Effort | **~3 engineer-days**, about 45 min of hands-on time with the drive live, no card time | ~11-13 engineer-days + 1 card job | ~14-16 | ~0.5 (a runbook) | 15-25+ (research) after the P2P scatter op exists |

**Recommendation (section 11):** **A now, before any customer KV is written** (plus D1 as the decommissioning step and `QWEN_PREFIX_NVME_DISCARD=1`). It closes the stolen-drive and decommissioned-drive cases at zero CPU cost, and it keeps P2P. **B only when a tenant deletion has to be cryptographic** (a deletion attestation, or a per-tenant zero-retention opt-out). Build B as C, on top of A: sealed objects take the host path, and an object is left unsealed for P2P only by an explicit class decision. D2 stays research until the P2P scatter op exists and a card window measures an ARX kernel.

**The drive itself:** the vendor's published specification for the cache drive (Samsung 9100 PRO, 1 TB) lists AES-256 encryption, TCG Opal and IEEE 1667 [V, vendor product page]. **The on-drive check is not done.** The drive's controller was offline when this was written: a bus re-enumeration after a card reset left the drive without its memory window. The tools were also not installed on the serving host. The read-only checks to run once the drive is live are in section 9.

## 1. What is on the drive, and the threats

### 1.1 Contents

The tier note covers this in sections 2, 3 and 9. Here is what an attacker holding the media can see.

* **Data region:** KV blocks of 2,228,224 B; GDN checkpoints of ~107-215 MB, with token ids and drafter window rows.
  * All of it is prompt-derived. It can in principle be turned back into text.
  * Objects are stored at 4 KiB-aligned LBAs, raw (no filesystem).
* **Index (snapshot and journal):** per object, the key (vLLM's chained block hash, salted per tenant), the chain id (also a salted hash), an 8-byte blake2b of the tenant tag, sizes, times, the sha256 of the object, and the layout fingerprint.
  * None of this is content. It is traffic metadata: who stored how much, and when.
* **Freed extents:** stay on the media until they are overwritten or discarded.
  * Deletion is logical and immediate in the index.
  * `QWEN_PREFIX_NVME_DISCARD=1` asks the drive to TRIM. A drive with deterministic read-after-TRIM then returns zeros through the OS, but the NAND may still hold the old pages until garbage collection.

### 1.2 Threats

| Threat | Meaning here |
| --- | --- |
| **T1 stolen drive** | the cache drive leaves the building alone, powered off; the attacker has a lab |
| **T2 decommissioned drive** | the drive is retired, returned under warranty, or fails; it may not accept commands any more |
| **T3 tenant deletion request** | a tenant (or the platform for it) asks that its retained state be gone *now*, not at the TTL |
| **T4 compromised host root** | an attacker is root on the serving host while it runs |
| **T5 operator** | someone with physical access (= T1/T2), or with root (= T4) |

**What no option here covers.**
* The engine holds every active tenant's KV in plaintext: on the cards, in the 48 GiB host tier and in the checkpoint store. Root on the serving host can read all of it.
* At-rest encryption protects the bytes *after* they leave the engine. The one thing a root attacker cannot undo is a key that was destroyed before the compromise.
* A stolen *whole host* that unlocks itself at boot (every unattended scheme below does) is not covered, except by network-bound unlock: the host fetches its unlock secret from a key server on its own network at boot, so a host taken off-site cannot unlock.

### 1.3 Shred granularity on this drive

**Expected: one namespace, no namespace management.** That is typical of client drives. It is **to verify** (section 9).

* **NVMe crypto erase, Opal `GenKey` and `Revert`.** All act on the namespace or the drive, so on this drive "per namespace" means the whole drive.
* **Opal locking ranges.** At least 8 are guaranteed (Opal 2), each with its own media key. But a range is a fixed LBA span.
  * The store has one allocator over one data region, for all tenants.
  * So per-tenant ranges would mean static partitions for at most 8 tenants. That does not fit "retention for all tenants" (section 6, D3).

## 2. Facts used

| Fact | Value | Source |
| --- | --- | --- |
| Cache drive | Samsung 9100 PRO 1 TB (PCIe Gen5 x4, running at Gen4), vendor: "AES 256-bit Encryption (Class 0), TCG/Opal, IEEE1667 (Encrypted drive)", 600 TBW / 5 years | [V] vendor product page; retailers and reviews say "TCG/Opal 2.0". Namespace count, sanitize and crypto-erase support are not published: section 9 |
| Drive read into host / into a card (P2P) | 6.9 / 6.95 GB/s at Gen4 | measured 2026-10-21 (tier note section 11) |
| 254k-token session | 3,841 blocks, 8.56 GB; 2.14 GB per card | tier note section 8 |
| Restore budget | ~1.2 s for a 254k session (the drive's own time at Gen4: 1.23 s) | [D] |
| Spill rate | ~416 GB/day stated; `through` ~420-460 GB/day | tier note section 8 |
| Today's digest | sha256 of every object at write; at restore `sample` (1 in 16), `all` or `off` | tier note section 3 |
| Serving host | x86 with AES-NI, VAES, VPCLMULQDQ, AVX-512 and SHA extensions, OpenSSL 3.5, a TPM 2.0 device, kernel built with `CONFIG_BLK_SED_OPAL=y` and `CONFIG_PCI_P2PDMA=y` | [V] read-only checks |

### 2.1 Cipher throughput (measured for this note)

`openssl speed -evp <cipher> -bytes 1048576 -seconds 2`, `nice -n 19`, pinned to one core with `taskset`.

* **Machine.** The measuring host is an Arm machine (Cortex-X925 big cores, Cortex-A725 little cores) running OpenSSL 3.0.13. It also serves and runs CI, so the runs were short and niced.
* **Not the serving host.** The tier runs on the serving host. Its x86 core with VAES and AVX-512 is expected to be at least as fast for AES-GCM (ESTIMATE).
* **Confirm before sizing.** Re-run the same seven 2-second commands there before sizing threads.

| Cipher (1 MiB buffers) | big core (X925) GB/s | little core (A725) GB/s |
| --- | --- | --- |
| AES-256-GCM encrypt | **8.10** | 2.60 |
| AES-256-GCM decrypt | **8.19** | 2.77 |
| AES-256-GCM encrypt, 16 KiB buffers | 8.08 | 2.59 |
| AES-256-XTS encrypt | 10.96 | 4.24 |
| AES-256-XTS decrypt | **10.98** | 4.23 |
| AES-256-CTR | 11.33 | - |
| ChaCha20 (stream only) | 3.04 | - |
| ChaCha20-Poly1305 | 2.15 | 0.82 |
| SHA-256 (today's digest, for comparison) | 2.49 | 2.09 |

[M] Each figure is one 2-second run. The host was under light load (load average ~2.5-3 on 20 cores).

### 2.2 What that means for the tier [D]

| Path | Volume | AES-256-GCM | AES-256-XTS (no integrity) | ChaCha20-Poly1305 | today: sha256 |
| --- | --- | --- | --- | --- | --- |
| 254k restore in 1.2 s (7.13 GB/s) | 8.56 GB | 1.05 core-s; **0.87 big core** (3 little) | 0.78 core-s; 0.65 big (2 little) | 3.98 core-s; 3.3 big (9 little) | `all`: 3.44 core-s; `sample`: 0.21 |
| Write-through, average | 4.8-5.3 MB/s | 0.06-0.07% of one big core (51-57 core-s/day) | less | 0.25% | 185 core-s/day (every object is hashed at write) |
| Spill at the host tier's measured full-tier rate | 5.6 GB/s | 0.69 big core (2.2 little) | 0.51 | 2.6 | 2.25 |
| One `through` step (32 blocks, 71.3 MB) | 71.3 MB | 8.8 ms of one big core, on the writer, never the scheduler | 6.5 ms | 33 ms | 29 ms |

How to read the table:
* **Restores.** The tier already reads on 4 threads, and decryption runs per 32 MiB run right after it lands, where the sha256 check runs today. A restore therefore keeps the drive's 1.23 s: the 1.05 core-s spread over 4 threads is ~0.26 s, and it overlaps the reads.
* **Writes.** **AES-256-GCM is cheaper than today's write-path sha256** (3.3x on this host). If B's authentication tag replaces the sha256, the writer's CPU goes *down* by ~70%.
* **XTS.** XTS is faster but authenticates nothing. It would keep the sha256 for integrity (0.78 + 3.44 core-s for a fully verified restore), and it binds every object to its LBA (section 4.6).

## 3. Option A: TCG Opal on the cache drive

**What it is.**
* The drive's AES engine encrypts every LBA with a media key, whether or not locking is enabled.
* Opal adds authentication: the media key of a locking range is released only after a PIN, and the range re-locks when the drive loses power.
* One locking range covers the whole namespace: the store's superblocks, snapshots, journal and data.

### 3.1 Coverage

* **Covered.**
  * **T1:** a powered-off drive is locked, and its contents are AES-256 ciphertext. That includes the index and every freed extent.
  * **T2:** a crypto-erase before the drive leaves; `Revert` or the PSID revert gives the drive a new media key. A drive that has died stays locked.
* **Not covered.**
  * **T3:** a deletion is logical, plus TRIM. Under Opal the remnants are ciphertext to anyone without the unlocked drive, but readable to the running host. A cryptographic deletion is whole-drive only.
  * **T4:** the range is unlocked to the OS.
  * **T5:** root.
* **Trust.** The protection is only as good as the drive's Opal implementation.
  * Published research (2018-2019) found self-encrypting drives from several vendors, older Samsung models among them, whose media key did not depend on the PIN.
  * The 9100 PRO's implementation has no public audit. What this rests on: the data is derived state with a 1-day TTL, and B can be layered on top (C).

### 3.2 Key management

* **Credentials.** The SID PIN and the Locking SP Admin1 PIN are random, 32 bytes each.
* **Where they live.**
  * Sealed by the serving host's TPM 2.0, so the root RAID's drives alone do not yield them.
  * Bound to the TPM, and to PCRs only if the boot chain is actually measured.
  * An offline escrow copy is kept by the owner.
  * The PSID printed on the drive's label is the last resort: a revert that destroys the data, which is fine for a cache.
* **Unattended unlock.**
  * A host unit runs before the serving container. It unseals the PIN and unlocks the range through the kernel's Opal ioctls, or through `sedutil-cli`, or through nvme-cli's `sed` plugin where the installed version has one.
  * A udev rule re-runs it whenever the drive's controller is (re)added. The drive has already dropped off the bus, once for power and once in a bus re-enumeration after a card reset.
  * If the unlock fails, the store's open fails: `nvme tier UNAVAILABLE` (tier note section 9). The engine serves on the host tier alone. This failure path is already built and tested.
* **Re-lock events.**
  * The range re-locks on a power cycle of the drive (`LockOnReset = PowerCycle`): a host power-off, or the drive's own power supply dropping.
  * A controller reset or a re-probe with the power kept does **not** re-lock unless `LockOnReset` includes them. Check this after setup.
  * Whether a warm reboot re-locks depends on whether slot power drops: test it.
  * A drive kept powered while it is moved to another machine stays unlocked (the "hot plug" attack). That needs physical access to a running host.
* **Rotation.**
  * The PIN changes at any time with `SetPIN`. It wraps the media key, so no data is rewritten.
  * Rotating the media key (`GenKey`) **is** a crypto-erase. It is the hardware form of `wipe()`.
* **Deploys and restarts.** Opal state belongs to the drive. A deploy does not touch it, so the tier survives deploys (owner decision: a deploy must not clear the tier).

### 3.3 Cost and P2P

* **Cost.**
  * Host CPU: none.
  * Throughput: none expected. The drive's AES engine is in the data path whether locking is enabled or not, so the 6.9 / 6.95 GB/s measurements already include it (ASSUMPTION, vendor-typical).
  * Confirm with the `nvme_store_bench.py` D0 job after enabling.
* **P2P: unchanged.** An unlocked range serves peer-to-peer reads like any other, and the card receives plaintext.

### 3.4 Effort and code

**Effort: ~3 engineer-days.**
* Setup and tests on the drive: 1 day.
* The TPM-sealed credential, the unlock unit, and the udev re-add path: 1-1.5 days.
* The runbook, including decommissioning (D1): 0.5 day.
* No card time. About 45 minutes hands-on with the drive live, best done in a window, since the gate formats the store anyway.

**Code (optional, +0.5-1 day).** The store format does not change. Two additions are worth making:

* **`QWEN_PREFIX_NVME_REQUIRE_SED=1`.** At open, `IOC_OPAL_GET_STATUS` must report Opal supported, locking enabled and unlocked; otherwise the tier refuses to start, with the same loud line and counter as the zero-retention refusal.
  * Without this a drive swap silently goes plaintext.
  * The ioctl needs `CAP_SYS_ADMIN`. The engine is root today; if it stops being root, the host unit can write a status file instead.
* **An `at_rest=opal|none` field** on the open line, and a gauge `nvme_at_rest`.

### 3.5 Operational steps

1. Run the read-only checks (section 9). Proceed only if Opal 2 is supported with a Locking feature, and record the locking range count.
2. Record the PSID from the drive label, offline.
3. Take ownership (SID), activate the Locking SP, and set the Admin1 PIN. Configure one locking range over the whole namespace (or the global range) with read and write locking enabled and `LockOnReset = PowerCycle`. Keep MBR shadowing off: this is not a boot drive.
4. Run `GenKey` on the range. This gives a fresh media key and destroys the P1 test patterns. Then run the gate profile with `QWEN_PREFIX_NVME_FORMAT=1`, which formats the store.
5. Seal the PINs with the TPM, and install the unlock unit and the udev rule.
6. **Test.**
   * Power-cycle the drive (or the host): the unit unlocks and the tier opens.
   * Disable the unit: `nvme tier UNAVAILABLE`, and the engine serves.
   * Re-probe the controller: the unit runs again.
7. **Decommissioning.** Run `Revert` (or the PSID revert), then a sanitize crypto-erase (D1) if the drive offers it. A drive that no longer answers stays locked; destroy it if policy demands.

## 4. Option B: software per-tenant keys on the host path

Every KV block and checkpoint is sealed with AES-256-GCM in the engine before it is written, under a data key (DEK) of its tenant. Restores open it in place in the slab slot the tier reserved.

### 4.1 Coverage

* **T1 and T2: covered, including a dead drive.** The ciphertext is useless without the KEK, which never touches the cache drive.
  * The index metadata (salted hashes, tenant hashes, sizes, times) stays readable on the media unless A is also on (C).
* **T3: covered per tenant (crypto-shred).**
  * Destroy the tenant's DEKs, then rotate the KEK (section 4.3). After that, no copy of the tenant's objects or remnants opens anywhere.
* **T4: not covered while the attacker is root.** The KEK and the resident DEKs are in engine memory.
  * But a tenant shredded **before** the compromise stays shredded. With A alone, root could read un-trimmed freed extents of deleted tenants.
* **T5:** physical = T1/T2, root = T4.

### 4.2 Key hierarchy

```
 KEK (node, 32 B)   held off the cache drive: a TPM NV index on the serving host (or a platform key service)
  |                 delivered to the engine container as a file; mlock'd, MADV_DONTDUMP, never logged
  +-- wraps --> DEK(tenant, generation) (32 B, random)   stored wrapped (AES-256-GCM, AAD = "qnvm-dek-v1" | tenant hash | salt hash | generation | store uuid)
                  |                                     in the store's journal (R_KEY) and snapshot (a key table)
                  +-- seals --> object (AES-256-GCM, 96-bit random nonce, AAD = store uuid | kind | key | tenant hash | chain | index | fp | nbytes | generation)
```

**Where the cache salt comes in.** The salt **selects** the key; it is never key material.
* `tenant_of_salt(salt)` gives the tag. Entries already carry `blake2b-8(tag)`, and the key table row is found by that and by a hash of the full salt.
* Do not derive a key from the salt alone. The salt is a request field that passes through the gateway and the engine. A key computed only from values like that can always be recomputed, so it can never be destroyed, so it cannot be shredded.
* Do not derive DEKs deterministically from the KEK either. To shred one tenant you would have to replace the KEK, and that would change every other tenant's DEK.
* Random DEKs, wrapped, are the structure that shreds one tenant at a time.

**Salt rotation.** Keying a DEK by the full salt (not only the tag) means a salt rotation starts a new DEK by construction.
* The old salt's blocks are unreachable anyway: every chain hash changes.
* The old DEK is destroyed when its last object leaves the index, within the 1-day TTL.

**Generations.** A tenant's DEK rolls daily (new writes use generation g + 1).
* A generation's DEK is destroyed when its last object leaves the index (by TTL, LRU or delete).
* So the remnants of expired objects become undecryptable at the next KEK rotation: a crypto-enforced cleanup of what the TTL already deleted logically.
* With per-tenant generations, the random 96-bit nonces are safe by a wide margin. Even a tenant that wrote the whole stated load (~1.9 x 10^5 objects a day) has a collision bound of ~2 x 10^-19 per generation.

### 4.3 Shred, rotation, reboot, power loss

* **Shred (`delete_salt` / `delete_tenant`).**
  1. Write the existing `R_DEL_TENANT` record and a new `R_KEY_DROP` record in one journal group.
  2. Write a new key table with every other DEK wrapped under KEK', and snapshot it.
  3. Destroy the old KEK only after the new superblock is durable.
  * Until step 3 the old wrapped DEK may still sit in the flash: NAND does not overwrite in place. After it, nothing opens.
  * Explicit deletion requests are done at once. Generation expiries are batched (e.g. hourly).
  * Each shred costs one snapshot, ~60 MB.
* **Where the KEK must live for a shred to be real.**
  * A TPM-*sealed* blob in a file is **not** enough. The TPM's storage key persists, so an old blob recovered from the root drives' flash still unseals.
  * Use a **TPM NV index** (32 bytes, overwritten in the TPM on rotation), or a sealing policy bound to an NV counter that is incremented on every rotation.
  * Keep two NV slots, so a crash between steps 2 and 3 leaves a KEK that opens the active table.
  * A platform key service works too. A database row has its backup retention as a shred gap, which the owner would have to accept.
  * The engine container must not touch the TPM. The KEK provider is a host unit with a small interface (get, rotate) over a Unix socket.
* **Reboot or power loss.**
  * The KEK survives in the TPM, and the key table survives in the journal.
  * The crash discipline extends the store's: an `R_KEY` record is durable before (or in the same journal group as) any PUT sealed under its generation. A PUT naming a generation whose key record did not survive is dropped at replay.
* **Deploys.** The KEK is delivered again and the key table replays, so nothing is lost. This holds only because the KEK is the host's, not the image's or the container's.
* **A lost KEK** (the TPM is cleared, or the board is swapped) leaves the store unreadable. For a cache that equals a format: drop all entries, then rotate.

### 4.4 Cost

* **CPU:** see section 2.2.
  * A 254k restore costs ~1.05 big-core-seconds of AES-256-GCM decryption, spread over the 4 existing reader threads and overlapped with the reads, so it adds nothing measurable to the 1.23 s (ESTIMATE).
  * The write path gets *cheaper* if the tag replaces the sha256.
* **Memory.**
  * **Writes cannot encrypt in place.** The slab slot is the host tier's live plaintext (with `through`, a held record is still served).
  * So each hash thread gets an aligned bounce buffer for its batch: at most 32 MiB per thread, 128 MiB for 4. Count it in the queue budget (tier note open question 3).
  * Restores decrypt in place: the destination is the tier's own reserved slot.
* **Index:** +32 B per entry (generation, nonce, tag). The snapshot budget goes from 200 to 240 B per entry: ~+12 MB of snapshot and of host RAM at ~370,000 entries.

### 4.5 P2P

**Incompatible for sealed objects.** On a P2P read the bytes go from the drive into card DRAM without touching the host CPU, so a host cipher cannot run on them. Section 7 asks whether the card could decrypt.

**What B costs in P2P value.**

* **Today: nothing.** P2P is a stub (`P2PBackend` raises `NotBuilt`), and every restore already takes the host path.
* **Once P2P exists:**
  * **The background restore.** A P2P restore is a background DMA plus a short scatter op. The host path holds the scheduler for the upload: ~0.96 s per 254k unsliced, which the sliced restore spreads but does not remove.
  * **Latency (ESTIMATE).** P2P at Gen4 with one drive, 1.23 s. The host path is ~1.3 s pipelined, ~2.2 s sequential. At Gen5 P2P is 0.61 s, or 0.31 s with two drives, while the host path stays at ~1-1.3 s, floored by the upload. So B gives up ~0.4-1.0 s per 254k restore at Gen5.
  * **Host resources.** 8.56 GB of host memory traffic and staging per 254k restore.
  * In the P2P design's own words, the case for P2P is that the restore stops blocking the engine, and the host RAM and CPU it frees. Sealed objects lose exactly that.

### 4.6 Why AES-256-GCM, not XTS

* **Integrity.** XTS has no tag, so the sha256 would stay. GCM authenticates every byte for less CPU than the sha256 costs today.
* **Location binding.** XTS's tweak is the LBA, so an object is bound to its location: a compaction or any move means re-encrypting. And a rewrite of the same LBA leaks equality per 16-byte block.
* **Format choice.** XTS suits a disk. The store is an object store with an index that has room for a nonce and a tag.
* **Not XAES-256-GCM.** XAES-256-GCM (a per-object key from a 192-bit nonce) is not among the cipher algorithms OpenSSL 3.0 or 3.5 lists [V], and per-tenant generations make the plain 96-bit nonce safe.
* **The library.** The standard library has no AES. CPython's `hashlib` normally links OpenSSL's libcrypto, so the image should already carry it (check); call `EVP_aes_256_gcm` through `ctypes`.
  * ctypes releases the GIL, so the existing reader and hash threads run in parallel, and `EVP_DecryptUpdate` works in place.
  * This keeps the tier free of non-stdlib Python dependencies. The `cryptography` package works too if the image has it; check that it drops the GIL on large buffers.

## 5. Option C: hybrid (Opal for the media, per-tenant keys on the host path)

A and B together.
* **Opal** covers everything on the media: the index, the journal, free space, and any object left unsealed.
* **B** seals objects per tenant.
* **Defence in depth.** If the drive's Opal implementation is weak, the sealed objects are still AES-256-GCM under keys that never touched the drive. If B's index metadata is the worry, Opal covers it.

**P2P under C.** P2P reads only objects written **unsealed**, by a class decision at spill time (`SpillKV` gets a `seal` flag). Candidate policies:
* tenants who opt into P2P latency and accept drive-level-only protection;
* or sessions above a size, where P2P matters most.

Unsealed objects are Opal-protected (T1, T2) but not shreddable per tenant (T3 logical only). Until P2P is built, the unsealed class is empty, and C costs what A + B cost.

**Effort:** A + B + 1-2 days for the class switch when P2P lands. **CPU:** B's, for sealed objects only.

## 6. Option D: other credible routes

* **D1. NVMe-native crypto erase only.**
  * Sanitize with the crypto-erase action (if SANICAP says so), or Format NVM with secure-erase setting 2 (if FNA says crypto erase is supported).
  * Seconds, no setup, no Opal configuration. The drive's always-on media key is replaced.
  * Covers `wipe()` and decommissioning of a **working** drive.
  * Does not cover T1 (no authentication: a stolen drive reads in the clear), a dead drive, or T3.
  * On a single-namespace drive the scope is the whole drive.
  * It needs NVMe admin commands, which the engine container must not issue: it is a host runbook step.
  * **Use it with A or B, never instead of them.** Effort ~0.5 day.
* **D2. B, decrypted on the card.**
  * Per-tenant keys *and* P2P: the P2P scatter op also decrypts.
  * The cipher has to be ARX (ChaCha20), because the card has no AES instructions (section 7).
  * The Poly1305 tag is checked only on host-path reads. On P2P reads the card checks a digest that sits in the authenticated index.
  * Host cost: ChaCha20-Poly1305 is 3.8x slower than AES-GCM on the measured host. A 1.2 s host-path 254k restore needs ~3.3 big cores (section 2.2), still affordable.
  * Effort 15-25+ engineer-days of research, after the P2P scatter op exists, plus card windows.
* **D3. Per-tenant Opal ranges or namespaces.**
  * Opal guarantees at least 8 ranges, and Configurable Namespace Locking gives a range per namespace.
  * On this drive: expected one namespace (to verify), and ranges are fixed LBA spans. So at most 8 statically sized tenants, and the store's allocator would split per range.
  * Not for "all tenants". It fits a future enterprise drive with namespace management, for a few large tenants: P2P-compatible per-tenant shred.
* **D4. dm-crypt / LUKS on the namespace.** Rejected.
  * Device-mapper does not pass the peer-to-peer flag, which kills P2P.
  * It is XTS with no integrity, under one key for all tenants, at B's CPU cost.
  * It gives nothing A + B does not.
* **D5. Ephemeral keys for zero-retention tenants (a variant of B).**
  * Their objects are sealed under a key that exists only in engine memory and dies with the container.
  * The drive then extends capacity within the container's life, and a restart or deploy loses those objects by construction.
  * Section 10 covers how this bears on the zero-retention conflict.

## 7. Can a Blackhole card decrypt?

**What the card has.** Each Tensix core has:
* five small RV32 cores that drive data movement and the math pipelines;
* a matrix FPU;
* the SFPU, a 32-bit vector unit (ASSUMPTION: 32 lanes).

The chip also has 16 RV64 cores (SiFive X280 class, in four clusters).

**No AES capability is documented** for any of them:
* no AES or GF(2^8) instructions;
* no carry-less multiply for GHASH;
* no cross-lane gather on the SFPU, so no table lookups.

**ARX ciphers fit the SFPU.**
* ChaCha20 needs only 32-bit add, xor and shifts (a rotate is two shifts and an or).
* The ttnn library this project builds exports `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_left_shift` and `bitwise_right_shift` [V: symbol names in the built library], and int32 `add` exists.
* So the primitives exist as SFPU operations. Speed needs one fused kernel: a chain of ttnn ops goes to DRAM per operation, orders of magnitude too slow.

**Cost estimate for ChaCha20 fused into the P2P scatter op (ESTIMATE).**

* **Operations per 64-byte block:** 80 quarter-rounds x 20 integer ops = 1,600, plus the feed-forward, the XOR with the data, and loads and stores, ~1,660. About 2,300 with the spills that 8 vector registers force on a 16-word state.
* **Ideal per Tensix core:** 32 lanes x 64 B per 2,300 instructions = 0.89 B/cycle, ~1.07 GB/s at 1.2 GHz.
* **Discounted 3-5x** for L1 traffic, issue overhead and pipeline latency: **0.2-0.35 GB/s per Tensix core.**
* **A 254k restore is 2.14 GB per card.**
  * On the full 130-core grid: **~47-82 ms** of device time per card (the cards work in parallel). Spread over the scatter chunks, that is about half to one decode round per 254k restore.
  * Alternatively, a reserved sub-grid of **5-9 cores** per card keeps pace with the drive (1.74 GB/s per card).
* **AES on the card.** Only bitsliced AES is possible: a ~113-gate S-box circuit, and ShiftRows/MixColumns as cross-lane moves under heavy register pressure, ~5-10x ChaCha20.
  * That is 0.25-0.8 s of full-grid time per 254k per card: several decode rounds per restore.
  * XTS decryption needs the inverse cipher, which is worse. GHASH on the card costs about as much again.
  * Not worth it. If the card decrypts, the on-drive format must be ChaCha20.
* **The X280 cores.**
  * There is no AES instruction set and no path to these cores in this project's stack.
  * Software table AES would run at ~0.1 GB/s per core, ~1.6 GB/s for all 16. That is at the edge of the 1.74 GB/s per card needed, before any data movement.
  * Rejected.
* **Key exposure.**
  * The per-object key goes to card L1 for the op. The card already holds every active tenant's plaintext KV, so this adds nothing new.
  * Integrity on P2P reads is the card-side digest, compared against the value in the authenticated index. It is not a MAC: a tamperer would need write access to the drive, which means host root or physical access.

**Verdict.** Plausible, unproven, and research. It depends on the scatter op (not built), on SFPU integer throughput that no card has measured for this, and on one card window for a microbenchmark before anyone commits to it.

## 8. Code changes for B (and what A needs)

**A** needs nothing in the store. The optional `REQUIRE_SED` check and the `at_rest` field are in section 3.4.

**B, `qwen_nvme_store.py` (the store stays a storage layer: no crypto in it):**

* **Entry v2.**
  * `VERSION = 2`. The unused `_flags` byte of `_ENTRY` gets bit 0 = sealed.
  * New fields: generation (u32), nonce (12 B), tag (16 B). Update `pack_entry`/`unpack_entry`; `DEFAULT_ENTRY_BUDGET` 200 -> 240.
  * A v1 store is opened as before; its entries are unsealed.
* **Journal and snapshot.**
  * New record kinds `R_KEY` (tenant hash, salt hash, generation, wrapped DEK, created) and `R_KEY_DROP`.
  * A key-table section in the snapshot, applied in `_apply_record`/`_decode_snapshot`.
  * `delete_tenant` writes `R_KEY_DROP` for every generation of the tenant in the same journal group as `R_DEL_TENANT`. `wipe` drops every key row.
  * Replay drops a PUT whose generation has no surviving key row.
* **`put_batch`.** Accepts sealed items (ciphertext buffers, nonce, tag, generation). `digest_of` runs only for unsealed items.
* **Tests.** The crash-at-every-device-call matrix (`CrashTests`) gains key records and a shred. New tests: a PUT whose key row was torn, the key table across a journal wrap.

**B, `qwen_kv_nvme.py` and a new `qwen_nvme_crypt.py`:**

* **`qwen_nvme_crypt.py`.**
  * An AES-256-GCM wrapper over libcrypto's EVP interface via `ctypes`: `seal_into(dst, src, key, nonce, aad) -> tag` and `open_in_place(view, key, nonce, aad, tag) -> bool`. It releases the GIL and makes no copy on open.
  * Known-answer tests with the NIST GCM vectors. A wrong key, a wrong AAD or a flipped bit must fail.
* **Flags (`parse`, contract).**
  * `QWEN_PREFIX_NVME_SEAL=off|tenant` (default off).
  * `QWEN_PREFIX_NVME_KEY_FILE`: the KEK, or the KEK provider's socket.
  * `RESTORE=p2p` is refused with `SEAL=tenant`, and `SEAL` is refused without the key.
  * Optional `QWEN_PREFIX_NVME_SEAL_ONLY=1` drops unsealed entries at open.
  * **The fingerprint does not change.** Sealing is per entry, so turning it on does not empty the tier, in keeping with the owner's rule that a deploy must not clear it.
* **`NvmeTier.__init__`.** Loads the KEK (mlock, `MADV_DONTDUMP`) and builds a key ring from the store's key table, unwrapping a tenant's DEK lazily on first use.
* **Writer: `_digests` becomes `_seal`, on the same hash pool.**
  * Per KV item: the DEK of `item.tenant` for the current generation, created with an `R_KEY` row if absent; a nonce from `os.urandom(12)`; the AAD; encryption into the thread's bounce buffer.
  * Checkpoints: `codec.pack`'s buffers are sealed as one object.
  * The sha256 is no longer computed for sealed objects.
* **Reader: in `read_into`, `verify` becomes `open_in_place` for sealed entries, and runs for every entry.** Authentication is part of decryption, so `QWEN_PREFIX_NVME_VERIFY` no longer applies to them.
  * A failed tag is reported exactly like a digest failure today: `'digest'`, `_forget`, a miss, a cold prefill.
  * `load_checkpoint` changes the same way.
  * `AUDIT` compares the ciphertext read back, plus one open.
* **Admin paths.**
  * `delete_salt`/`delete_tenant`: the store delete, the key drop, then a KEK rotation through the provider.
  * `wipe`: drop all keys, then rotate.
  * If no provider can rotate (a plain key file), a shred is logical plus a destroyed key row: complete only at the next rotation. The log line must say so.
* **Metrics and lines.**
  * Counters: `nvme_sealed_{objects,bytes}`, `nvme_open_failures`, `nvme_keys_{created,destroyed}`, `nvme_kek_rotations`; `at_rest=opal|tenant|none` on the open line.
  * Never a key, a salt or a tenant tag in a line.
* **`P2PBackend` docstring.** Sealed objects are not eligible; only the unsealed class (C) or a card-side cipher (D2) is.

**Effort for B: ~11-13 engineer-days.**

| Item | Days |
| --- | --- |
| crypt module + KATs | 1.5 |
| key table, journal records, crash tests | 3 |
| writer seal path + bounce buffers | 1.5 |
| reader open path, checkpoints, audit | 1.5 |
| shred + rotation | 1 |
| KEK provider (host unit, TPM NV, two slots) | 2 |
| contract, flags, markers, metrics | 1 |
| review | 1 |

Then one card job: N1 exactness and N2 timing with `SEAL=tenant`, ~2 h.

A production **deletion path** is missing in every option. Today only the gate's command directory and an admin thread call `delete_tenant`. The platform needs a route that reaches it (+1-2 days, platform side).

## 9. The Opal check on the actual drive: to verify once the drive is live

**Status: not run.**
* When this was written the cache drive's controller was offline. The kernel had disabled it after a card reset's bus re-enumeration released its memory window, so no command of any kind could reach it.
* `nvme-cli`, `sedutil-cli` and `smartctl` were not installed on the serving host, and nothing was installed for this note.
* The evidence today is the vendor's (section 2).

**This incident matters to A's design.** A card reset that re-enumerates the switch can take the cache drive with it. The unlock unit must re-run after every re-probe, and the tier's latch-and-serve path is what keeps the engine up meanwhile.

**Read-only commands, in order.** None writes to the drive.
* Identify, Security Receive and Get Log Page only.
* `IOC_OPAL_GET_STATUS` returns flags the kernel keeps. `IOC_OPAL_DISCOVERY` performs a Level 0 Discovery, which is a Security Receive.
* `X` is the cache drive's controller number, which can change after a re-probe.

```
# 0. which controller is the cache drive, and is it live (state must be "live", size non-zero)
cat /sys/class/nvme/nvme*/model /sys/class/nvme/nvme*/state
lsblk -o NAME,MODEL,SIZE,LOG-SEC,PHY-SEC

# 1. no package needed: the kernel's Opal status (CONFIG_BLK_SED_OPAL=y). IOC_OPAL_GET_STATUS = _IOR('p', 236, 8 bytes) = 0x800870EC
sudo -n python3 -c 'import fcntl,os,struct; fd=os.open("/dev/nvmeXn1",os.O_RDONLY); b=bytearray(8); fcntl.ioctl(fd,0x800870EC,b); f=struct.unpack("<I",b[:4])[0]; print(hex(f),[n for n,v in (("supported",1),("locking_supported",2),("locking_enabled",4),("locked",8),("mbr_enabled",16),("mbr_done",32),("sum_supported",64)) if f&v])'

# 2. with nvme-cli (needs installing; the owner's call)
sudo -n nvme id-ctrl /dev/nvmeX -H | grep -E -A16 '^(oacs|fna|sanicap) '      # OACS bit 0 security send/receive, bit 1 format, bit 3 namespace management; FNA bit 2 crypto erase; SANICAP bit 0 crypto erase
sudo -n nvme id-ctrl /dev/nvmeX | grep -E '^(nn|mdts|oncs|vwc|oacs|fna|sanicap) '
sudo -n nvme id-ns /dev/nvmeX -n 1 -H | grep -E 'nsze|LBA Format|in use'     # 512 B or 4 KiB in use; a 4 KiB format offered?
sudo -n nvme security-recv /dev/nvmeX --secp=0 --spsp=0 --size=256 --al=256  # supported security protocols (0x01 = TCG, 0xEE = IEEE 1667)
sudo -n nvme security-recv /dev/nvmeX --secp=1 --spsp=1 --size=2048 --al=2048 # TCG Level 0 Discovery, raw
sudo -n nvme get-log /dev/nvmeX --log-id=0x81 --log-len=20                    # sanitize status
sudo -n nvme smart-log /dev/nvmeX                                             # wear baseline before the tier writes

# 3. with sedutil-cli (if installed)
sudo -n sedutil-cli --scan                     # the drive should show Opal 2
sudo -n sedutil-cli --isValidSED /dev/nvmeX
sudo -n sedutil-cli --query /dev/nvmeX         # NOTE: also sends a Properties call via Security Send (changes no state); skip it if any security-send is ruled out
```

**What each answer decides.**

| Read | Decides |
| --- | --- |
| Opal supported (`GET_STATUS` bit 0; Level 0 feature 0x0203 Opal SSC 2) | A is possible |
| Locking feature (0x0002): supported, enabled, locked; media encryption bit | the starting state; whether anyone set it up before |
| Opal SSC 2 descriptor: number of Locking SP admin/user authorities | how many ranges D3 could have |
| Single User Mode (0x0201), Configurable Namespace Locking (0x0403) | whether per-range owners or per-namespace ranges exist |
| `nn`, OACS bit 3 | one namespace or several (D3) |
| FNA bit 2, SANICAP bit 0 | D1 by Format or by Sanitize |
| LBA format in use | the store assumes 4 KiB alignment; a 512 B format works but a 4 KiB one is cleaner (a reformat erases) |

## 10. Zero retention and crypto-shred claims

**The conflict** (tier note section 9). A node that advertises zero retention signs, for every inference container it destroys, that nothing survived. The tier keeps prompt-derived KV for a day after the container is gone. The tier refuses to start where it sees that signal. **The platform policy is the owner's call; what each option lets the platform say:**

* **A.**
  * A whole-drive crypto-erase is a real shred of everything. It supports a decommissioning claim ("the drive was crypto-erased") and an "at rest, encrypted by the drive" claim.
  * It **does not** make "nothing survived this container" true. KV survives the container by design; Opal only hides it from someone without the unlocked drive.
  * A zero-retention node could run the tier under A only if it crypto-erased the drive at every container teardown, which defeats retention.
* **B.**
  * Enables a **per-tenant deletion attestation**: tenant X's retained state was crypto-shredded at time T, and its keys are destroyed.
  * With a per-tenant opt-out (no DEK, no writes for zero-retention tenants), a node could truthfully say "nothing of *this* tenant was retained", while retaining others'.
  * The owner's current policy is retention for all tenants, so B does not resolve the conflict today. It makes a future per-tenant policy checkable.
* **C.** As B for sealed objects. Objects left unsealed for P2P can carry only A's claims.
* **D5 (ephemeral keys for zero-retention tenants).** Their ciphertext may remain on the media, but no key for it survives the container.
  * Whether the attestation may count that as "nothing survived" is a wording question for the platform and the owner. It is the strongest technical form of the claim a drive-backed tier can make.
* **D1.** Supports decommissioning claims only.

## 11. Recommendation

1. **Now, before customer KV lands on the drive: A**, with `QWEN_PREFIX_NVME_REQUIRE_SED=1` so a drive swap cannot silently go plaintext. Also `QWEN_PREFIX_NVME_DISCARD=1`, and D1 in the decommissioning runbook.
   * It covers the stolen-drive and decommissioned-drive cases, which are the at-rest threats encryption can actually change here.
   * It costs no CPU, does not change the store format, and keeps P2P, the owner's stated direction.
   * It takes ~3 engineer-days and about 45 minutes with the drive live.
   * It rests on trust in the drive's Opal implementation, which is acceptable for derived state with a 1-day TTL.
2. **B when the product needs per-tenant cryptographic deletion**: deletion attestations, or per-tenant zero-retention opt-outs.
   * Build it as C, on top of A, ~11-13 engineer-days.
   * Its CPU cost is small: ~1 big-core-second per 254k restore, less than today's sha256 on the write path.
   * Its real cost is that sealed objects can never take the P2P path.
3. **P2P with per-tenant keys (D2)** only after the P2P scatter op exists, and only after one card window measures a ChaCha20 SFPU microbenchmark at or above ~0.2 GB/s per core. AES on the card is not worth pursuing.

Why not B now:
* P2P is the reason the tier stores raw namespace LBAs. B forecloses it for every sealed object, and per-tenant shred buys little that the 1-day TTL, an immediate logical delete, TRIM and Opal do not already give, unless someone must *prove* a deletion.
* That is a product decision, not an engineering one.

## 12. Open questions for the owner

1. **A now?** If yes: the window to set it up (the drive must be live and is formatted anyway), the PIN escrow, and whether `REQUIRE_SED` is mandatory in traffic profiles.
2. **Per-tenant cryptographic deletion:** a requirement (B / C), or is logical delete + TRIM + Opal + the 1-day TTL enough?
3. **Where the KEK lives** for B: a TPM NV index on each serving host (local, no backup gap), or a platform key service (central audit, a backup-retention shred gap)?
4. **Under C, which objects may stay unsealed for P2P:** none until asked, per-tenant opt-in, or by session size?
5. **Trust in the drive's Opal implementation:** the same decision the platform's cold-tier design lists as an owner decision.
6. **The drive dropping off the bus at card resets:** a reservation of memory-window space for the switch's ports, a different slot, or a recovery step in the card-reset runbook. Any of them must be followed by the unlock unit.
