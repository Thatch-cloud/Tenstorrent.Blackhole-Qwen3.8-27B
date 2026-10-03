# Two-card recipes: Qwen3.8-27B on two Blackhole p150a cards

> **Not runnable from a clean clone yet.** The image these commands start is stage 8 of [BUILD.md](BUILD.md), and
> stages 7-8 depend on a serving runtime tree that is not published. Stages 1-6 (tt-metal, the upstream PR stack and
> the five custom ops) are reproducible; read "What is not here yet" in BUILD.md before planning around any recipe.

Every recipe here serves `Qwen/Qwen3.8-27B` on **two p150a cards as a 1x2 mesh** (`MESH_DEVICE=P300`, the name
upstream tt-metal gives that shape) through the same image, selected by one profile name. This page says what each
profile is, what was measured, how to run it and where it stops. [BUILD.md](BUILD.md) builds the image.

Where a number comes from a document in this repository, the document is named. Figures from the maintainers' own
gate runs that have no public document are not quoted here; measure on your own cards.

## Which recipe

| Profile | In one line | Context | Seats | Links needed | Binary tier |
|---|---|---|---|---|---|
| `general-2link` | stock decode, any request; the one that opens with two trained links | 65,536 | 4 | 2 | `k64j` (`ops` unmeasured) |
| `general` | the same, under the standard `p150_x2` descriptor | 65,536 | 4 | 4 | `k64j` (`ops` unmeasured) |
| `general-prefix` | `general` plus conversation prefix reuse | 65,536 | 4 | 4 | `k64j` |
| `exact` | speculative decoding, 4 users at 131,072; frozen benchmark shape | 131,072 | 4 | 4 | `k64j` |
| `c2` | `exact`'s geometry serving any request | 131,328 positions | 4 | 4 | `k64j` |
| `c2-packed` | `c2` with packed rounds at any position | 131,328 positions | 4 | 4 | `k64j` plus local evidence |

Start with `general-2link` (two links) or `general` (four links). The speculative-decoding profiles need the DFlash2
draft fixtures, four trained links and a binary hash pinned to your own build, and the last three have the limits
listed below.

**Not included:** the sticky-session and parked-engine profiles (no two-card hardware pass is recorded), the gate-only
profiles (they exist to qualify an image), the `coding` benchmark profile (not servable), and the four-card profiles.

## Common to every recipe

- **Hardware:** two p150a on one host, firmware 19.12.0, two free 1 GiB hugepages (`/dev/hugepages-1G`). Check the
  card-to-card links with `test_system_health` (built in stage 1 of the chain): the profiles named "4 links" refuse to
  open on a pair that trained fewer, by design (`STRICT_INIT`).
- **Model:** `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` in a Hugging Face cache mounted
  at `/models`. The mount must be writable: the converted-weight cache and the kernel cache live in `/models/.qwen-c2`,
  and the first start converts the weights.
- **The profile selects everything.** `QWEN_C2_PROFILE=<name>` makes the image's boot hook apply that profile's engine
  settings, mesh descriptor and environment over the command line (`scripts/ci/qwen_c2_profiles.json`,
  `scripts/ci/serving_c2_contract.py`).
- **Image:** the serving image of [BUILD.md](BUILD.md), section 6, tagged `localhost:5000/qwen-c2-serving:local`.

```bash
# Two board ids from /dev/tenstorrent/by-id/ (list the directory; do not guess).
BOARD0=<board-0>  BOARD1=<board-1>
PROFILE=general-2link               # or general, general-prefix, exact, c2, c2-packed
HF_HUB_CACHE=/path/to/hf-hub-cache   # holds models--Qwen--Qwen3.8-27B; writable

docker run -d --name qwen-two-card --read-only \
  --tmpfs /tmp:rw,size=512m --tmpfs /opt/tt-metal/generated:rw,size=1g \
  --tmpfs /root/.cache/ttnn:rw,size=1g --tmpfs /root/.cache/tt-metal-cache:rw,size=8g \
  --shm-size 4g --memory 80g --cpus 8 \
  --device "$(readlink -f /dev/tenstorrent/by-id/$BOARD0)" \
  --device "$(readlink -f /dev/tenstorrent/by-id/$BOARD1)" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HF_HUB_CACHE":/models -e HF_HOME=/models -e HF_HUB_CACHE=/models \
  -e MESH_DEVICE=P300 -e QWEN_C2_PROFILE="$PROFILE" \
  -p 127.0.0.1:8000:8000 \
  --entrypoint python3 localhost:5000/qwen-c2-serving:local \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice

python3 scripts/ci/c2_serving_smoke.py http://127.0.0.1:8000     # stdlib only: liveness, 4 concurrent streams, agreement, a short benchmark
```

The first start takes a long time (weight conversion, then kernel compilation); later starts reuse
`/models/.qwen-c2`. `SYS_NICE` silences the hugepage NUMA warning. The container is read-only on purpose: everything
it writes is in the tmpfs mounts and in `/models`.

After any change of image or cabling, read `docker logs qwen-two-card` first: a link refusal, a missing fixture or a
binary-hash mismatch stops the start-up and is reported there.

---

## 1. `general-2link`

`general` under a two-channel 1x2 mesh descriptor (`scripts/ci/qwen_p150x2_2link_mesh_graph_descriptor.textproto`; the
image places it at `/opt/qwen-c2/mesh/`). It is for pairs whose cards train **two** Ethernet links between them. The
standard `p150_x2` descriptor declares four channels per edge, and the plugin refuses a pair that trained fewer.

- **Run:** `PROFILE=general-2link`.
- **Results:** no published figure yet. This is the only recipe the maintainers can re-check on their current cabling.
- **Limits:** 65,536 tokens of context, four seats, no speculative decoding, sampling and long outputs as in stock vLLM.
  The profile's own description in `qwen_c2_profiles.json` still calls it unverified; that text is stale.

## 2. `general`

The image with the speculative fast path off: the TT plugin's stock decode with the image's kernels and the model
graft. Any request shape (sampling, long outputs, any prompt up to the 65,536 window).

- **Run:** `PROFILE=general`.
- **Results:** see `docs/tp4-g1-bringup.md`, "What speed to expect".
- **Limits:** 65,536 context, four seats. Needs **four trained links**. As configured the image sets
  `QWEN_FAST_SDPA_PF=1` for every profile, which needs the `k64j` binary tier; an `ops`-tier build must set
  `QWEN_FAST_SDPA_PF=0` (`-e QWEN_FAST_SDPA_PF=0`), a configuration the maintainers have not measured.

## 3. `general-prefix`

`general` plus prefix reuse: vLLM's attention prefix cache and GDN-state checkpoints saved at 2048-token boundaries, so a
turn that extends an earlier conversation does not recompute it.

- **Run:** `PROFILE=general-prefix`, plus a salt key (below).
- **Results:** no published figure yet; the saving is the time to first token of a continuation.
- **Salt key:** reuse is keyed by a per-tenant `cache_salt` that the image verifies. Create a key file of at least 32
  random bytes at `/models/.qwen-c2/prefix-salt.key` (inside the cache mount), then mint a salt for each client:
  ```bash
  docker exec qwen-two-card python3 -c "from serving_c2_contract import mint_salt; \
    print(mint_salt(open('/models/.qwen-c2/prefix-salt.key','rb').read(), 'tenant-abc12345'))"
  ```
  and send it as `cache_salt` in the request body. Without the key every salt is dropped and the profile serves exactly
  as `general`. The image polls for `/models/.qwen-c2/prefix-reuse.off`; creating it switches reuse off (it latches) without a restart.
- **Limits:** as `general`; needs four trained links.

## 4. `exact`

Speculative decoding (DFlash draft, 15 speculative tokens) for four users with a 131,072-token context and a 256-token
output budget: the shape the maintainers' real-text gate measured.

- **Run:** `PROFILE=exact`, an image built with the DFlash2 fixtures (BUILD.md, "Fast tier").
- **Results:** exact against single-stream decoding. Public context: `docs/four-streams-131k-feasibility-2026-09-23.md`
  ("about 28.3 tok/s per user") and `docs/real-text-2026-09-24.md` (real-text single stream 32.5 tok/s at 131k, about
  20.7 tok/s per user at four on the earlier image).
- **Limits:** **it admits only prompts of exactly 131,072 tokens**, so it is a benchmark shape, not a server. It needs
  the frozen-evidence tree (now in `bundle/`), four trained links and `QWEN_FAST_RUNTIME_BINARY_SHA256` set to your
  build's hash (the image takes it as a build argument). **On a source-built `k64j` tree it is refused at attach**:
  the binary override and the T16 evidence pin the SDPA prefill factory at `fd8c0676...`, and the k64j build leaves
  `bfab8558...` on disk. No variable or build argument changes that; BUILD.md, "Pins that `exact`, `c2` and
  `c2-packed` check", has the table and the two ways forward. Only the `general*` profiles avoid it.

## 5. `c2`

`exact`'s geometry serving any request: prompts up to 123,136 tokens, outputs up to 16,384.

- **Run:** `PROFILE=c2`; same requirements as `exact`, **and the same pin refusal on a source-built `k64j` tree**
  (BUILD.md, "Pins that `exact`, `c2` and `c2-packed` check").
- **Results:** no published figure yet.
- **Limits:** below `general` under concurrency (the maintainers' own measurements say so). The profile's description still says "not qualified for production
  traffic".

## 6. `c2-packed`

`c2` plus packed rounds: four users advance together at any position, using the runtime-extent decode flag (0x20) of
the K64j binary tier.

- **Run:** `PROFILE=c2-packed`, on the `k64j` tier only.
- **Results:** no published figure yet.
- **Limits:** the admission check pins the maintainers' binary, four kernels and its evidence record, and it also pins
  the decode factory the k64j build changes. **On a binary you built, the attach is refused until you run the evidence
  checks on your own cards and re-pin, and a refusal stops the engine from starting** (by reading the code: it does not
  fall back to `c2`; an earlier version of this page said it did). The evidence harnesses and re-pin procedure are not
  in this repository yet (BUILD.md, "What is not here yet").

---

## Limits that apply across the table

- **Binaries are not bit-identical across builds.** Your `_ttnncpp.so` will not hash like the maintainers'. The
  acceptance for a rebuild is `tt-metal-custom-ops/verify.sh` (exact marker set, bindings), the unit tests that run
  inside the image, and `c2_serving_smoke.py`; not a hash comparison.
- **Four-link profiles on a two-link pair do not open.** If `test_system_health` shows two links, use `general-2link`.
- **A real P300 board is untested.** The recipes were developed on two separate p150a cards. If you have a P300, the
  maintainers would like to know what the link count and firmware report.
- **`decode_gated_delta_rule`** ships in the version the maintainers' build container used; which kernel version their
  production binary ran is not settled (see `tt-metal-custom-ops/README.md`, "Open points").
- **Numbers on this page** are only those tied to a named document in this repository.
