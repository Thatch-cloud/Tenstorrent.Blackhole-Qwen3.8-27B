#!/usr/bin/env bash
# Build ~/opgraft-K64j on the rig HOST: ~/opgraft-K64i (the served graft) plus the K64j decode factory (the runtime
# extent, flag 0x20: apply_factory_k64j.py F19-F22 over K64i's stage 4) and its four kernels (make_k64j_kernels.py
# R10, C3, W4), through the ttbuild container. These are the build steps 2-7 of ../k64j_probe/README.md ("K64j
# itself: the edit list and the build"), modelled step for step on ../sdpa_decode_slice/build_k64i.sh. Step 1 (ship
# over plink) is the checkout itself. It opens no device and launches no device container.
#
#   K64j = K64i's contents (attn_prep, nlp_concat_heads_decode, sdpa_decode, sdpa, ...)
#        + _ttnncpp.so / _ttnn.so rebuilt in ttbuild over BOTH factories: the decode factory at K64j (bb4dc6a7,
#          apply_factory_k64j.py over the saved 3e0a69af) and the prefill factory exactly as K64g / K64i built it
#          (apply_factory_pf.py, bfab8558), so [QWEN-SDPA-PF] survives
#        + the four K64j kernels over K64i's in sdpa_decode/device/kernels: dataflow/reader_decode_qwen.cpp,
#          dataflow/reader_decode_qwen_slice.cpp, compute/sdpa_flash_decode_qwen.cpp and
#          dataflow/writer_decode_qwen_slice.cpp. sdpa_decode's factory .cpp stays the audited 3e0a69af (the compiled
#          factory lives in the .so), and no stock kernel or shared header changes.
#
# A graft .so replaces the WHOLE binary (memory graft-so-drops-image-patches), so step 6 checks that every distinct
# QWEN_ / [QWEN- string of K64i's _ttnncpp.so AND of each image's is still in the new one, plus the K64j literals.
#
# THROUGH CI (the user is offsite; docs/card-b-runner.md): .github/card-b-job.env
#   CARD_B_HARNESS=optimisation/ttnn-op/k64j/build_k64j.sh
#   CARD_B_ARGS=
#   CARD_B_ENV=K64J_BUILD_DRY_RUN=1      # first: every check the host can make, and every command, nothing run
#   CARD_B_ENV=                          # then the build: two ninja runs of about 30 s, the image copies
# The card-B runner runs it on the rig host from the checkout root, with QUAL_CARD (card B), RESULTS
# (/tmp/card-b-results/<run id>) and CARD_B_ARGS (must stay empty); the console, MANIFEST.sha256, the QWEN-string
# lists and a summary come back in the card-b-<run id> artifact. Record the K64J_TTNNCPP_SHA256 line: the card tests
# take it as EXPECT_TTNNCPP_SHA256 (k64j/run_card_b.sh), the arm as QWEN_FAST_RUNTIME_BINARY_SHA256.
# BY HAND on the rig: bash optimisation/ttnn-op/k64j/build_k64j.sh [graft dir]
#
# Env (CARD_B_ENV carries NAME=value words without spaces; K64J_IMAGES therefore also takes commas):
#   K64J_BASE_GRAFT        the base graft (~/opgraft-K64i); K64J_BASE_SHA256 its _ttnncpp.so (K64i's cf54d716...;
#                          empty skips)
#   K64J_GRAFT             the graft to write (~/opgraft-K64j; a first argument also sets it); built in <it>.partial
#                          and moved into place only after step 6
#   K64J_DECODE_BASE_DIR   the saved 3e0a69af decode factory (~/kwork64/k64f, where K64f/K64i left it)
#   K64J_PF_BASE_DIR       the saved fd8c0676 prefill factory (~/kwork64/k64g); either is saved from ttbuild when
#                          ttbuild holds the base
#   K64J_WORK_ROOT         work-<stamp>/ (backups, copies out of ttbuild, strings lists): ~/kwork64/k64j, never the
#                          checkout (the next checkout must be able to clean the workspace)
#   K64J_IMAGES            images whose sdpa_decode/ and sdpa/ the graft's must equal but for the qwen kernels, and
#                          whose _ttnncpp.so QWEN strings must survive (default: P8 57cb6994, the card tests' and the
#                          C2 arm's image; a subset must keep it); K64J_SKIP_IMAGE_COMPARE=1 (last resort)
#   K64J_KEEP_STAGED=1     leave ttbuild on the qwen factories and kernels (a rerun then patches from the saved bases)
#   K64J_BUILD_DRY_RUN=1   print every docker, cp, mv, rm, touch, mkdir and generator command as '### run: ...' and
#                          run none of them; the read-only checks of the checkout, the base graft and the saved bases
#                          (when they exist here) still run
#   QUAL_CARD              only validated (qual_card_select): this build never opens a card
#
# ttbuild's sources are left as they were found (decode factory 3e0a69af, prefill factory fd8c0676, no qwen kernel
# files) unless K64J_KEEP_STAGED=1; everything replaced is saved under work-<stamp>/backup. build_Release is rebuilt
# from the restored factories (step 7, run before step 6 as build_k64i.sh does) and checked to have lost every qwen
# branch: docker cp keeps the saved files' OLD mtimes, so without a touch and a rebuild ninja would call the unity TUs
# up to date. On a failure the EXIT trap restores and touches the sources but does not rebuild; it says so.
set -euo pipefail
# Byte-order collation for every sort/comm (the K64g lesson: comm under one locale, sort under another).
export LC_ALL=C

# ---- recorded shas (test_build_k64j.py keeps these equal to the Python and to build_k64i.sh) ----
K64I_TTNNCPP=cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4
FACTORY_BASE=3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a
FACTORY_UNPATCHED=05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792
FACTORY_QWEN_STAGE1=1b54abd3fe466a058046939e2ec0366505f80fde631dba387c591985fa3da301
FACTORY_QWEN_STAGE3=06167779a979ba1f35c78d1c002f9956215ca52a4ccd4531e0f5f5fe4e44191d
FACTORY_QWEN_STAGE4=1634369677ae0247a387abb5610b8ee19f2b7a87d476534dd1e05bf6714529e3
FACTORY_K64J=bb4dc6a759d40054d31089792f617a1d66f5837da2065fa12f61438d09a55080
PF_BASE=fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783
PF_FACTORY=bfab8558d889ad215f0e9ee732c75a4142be1a1e7f37810f4ca8c5e3c73bdf65
PF_READER_BASE=f97f5490cf476db92d575de33c85f8707f96d7474896ee086ee864c23a3efa27
PF_READER=eecc1166a209e61dc8498149b5a4338cc278d7106f4ca68d6434f620e942f8d8
READER_ALL=49a05926b437e2ca90d7e01c60e85a6b11f333375a6159fa02c9ff6f78af764e
COMPUTE_ALL=d24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f
WRITER_ALL=734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8
DATAFLOW_COMMON=e4623a2254559eaec4450ebfab0f9c5732e02acfe4d8126bd5eeb7efe0fdc608
RT_ARGS_COMMON=1b52c60d78ada6f08effd326c2ed2407b3a74cf0db2353fadbe51b088610aec8
READER_QWEN=280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b
COMPUTE_QWEN=8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a
READER_SLICE=0f5a019ccc06ca603bb4ed44c77cc66f9cb3e5bd35193810eb127f13c9f5631f
WRITER_SLICE=ac6cf815c34df85a9d39593d95f28eb2da0cb5b37c232633e65bf3ed116925f4
K64J_READER_QWEN=adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49
K64J_READER_SLICE=518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a
K64J_COMPUTE_QWEN=409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e
K64J_WRITER_SLICE=642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a
GATE_IMAGE=sha256:57cb699489436842d7e7bdd5ab917d509b4492fc95f6f083ef7d2865b488c2ef   # P8: the card tests' and the C2 arm's
SHARE_MARKER='[QWEN-SDPA] KV-share twin bands'
STAGE1_REFUSAL='[QWEN-SDPA] KV share is not in this build'
SLICE_MARKER='[QWEN-SDPA] q-slice rows_per_kv='
EXTENT_MARKER='[QWEN-SDPA] runtime-extent entries='
STAGE4_MARKERS=("$SLICE_MARKER" '[QWEN-SDPA] KV read-ahead needs KV share' '[QWEN-SDPA] q-slice saves no tile'
                '[QWEN-SDPA] q-slice does not cover KV head' reader_decode_qwen_slice.cpp writer_decode_qwen_slice.cpp)
K64J_MARKERS=("$EXTENT_MARKER" '[QWEN-SDPA] runtime extent (0x20) needs the tail flag (0x1) and a narrow'
              '[QWEN-SDPA] runtime extent (0x20) needs an interleaved cur_pos tensor'
              '[QWEN-SDPA] runtime extent (0x20) needs an int32 row-major cur_pos tensor of B='
              '[QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor')
PF_MARKERS=('[QWEN-SDPA-PF] flags=' '[QWEN-SDPA-PF] kv_chain outside its qualified envelope' 'reader_interleaved_qwen_chain.cpp')
# The four K64j kernels, by their path under sdpa_decode/device/kernels (the layout of k64j/kernels too).
DECODE_KERNELS=(dataflow/reader_decode_qwen.cpp dataflow/reader_decode_qwen_slice.cpp compute/sdpa_flash_decode_qwen.cpp
                dataflow/writer_decode_qwen_slice.cpp)
k64j_sha() {  # the recorded sha of a K64j kernel path
  case $1 in
    dataflow/reader_decode_qwen.cpp) echo $K64J_READER_QWEN ;;
    dataflow/reader_decode_qwen_slice.cpp) echo $K64J_READER_SLICE ;;
    compute/sdpa_flash_decode_qwen.cpp) echo $K64J_COMPUTE_QWEN ;;
    dataflow/writer_decode_qwen_slice.cpp) echo $K64J_WRITER_SLICE ;;
  esac
}
k64i_sha() {  # the recorded sha of the K64i kernel it replaces
  case $1 in
    dataflow/reader_decode_qwen.cpp) echo $READER_QWEN ;;
    dataflow/reader_decode_qwen_slice.cpp) echo $READER_SLICE ;;
    compute/sdpa_flash_decode_qwen.cpp) echo $COMPUTE_QWEN ;;
    dataflow/writer_decode_qwen_slice.cpp) echo $WRITER_SLICE ;;
  esac
}

# >>> qual_card.sh: which board a qualification harness runs on (canonical copy scripts/ci/qual_card.sh)
# Every single-card harness under optimisation/ttnn-op embeds this block byte for byte (the scripts in
# scripts/ci source the file); scripts/ci/test_qual_card.py fails when a copy drifts, and
# `py -3.11 -B scripts/ci/test_qual_card.py --sync` re-copies the canonical text into every harness.
#
# The rig has three p150a. Card M (blackhole-CEF5729692C19E6D) and card A (blackhole-3707293C249A5E67)
# are the serving pair: Ethernet-linked, mounted by every CI gate arm (lever_n_m3native_run_arm.sh), and
# reset together by the gate. Card B (blackhole-F36F768B9A5CAFA0, PCIe only) is the qualification card.
# /dev/tenstorrent/N numbers change across resets and switch power-cycles, and tt-smi's own board index
# is a different numbering again, so nothing here hard-codes either: the target is a board id, its node
# is resolved with readlink -f at launch and again right before the container starts, and the reset
# hint prints commands that resolve the board id when they are run (never a bare index) plus the PCI
# address that identifies the board's row in tt-smi -ls.
#
# Only the m3native gate (qwen-lever-n-m3native-gate.yml) is scoped to the serving pair. qwen-card-reset.yml
# and most other qwen-* hardware workflows still act on every node, or on fixed node numbers, in the same
# qwen-two-p150a-exclusive group: check gh run list for that group before and during a card-B session.
#
#   QUAL_CARD=<board id>    the target, a name under /dev/tenstorrent/by-id (default: card B)
#   ALLOW_SERVING_CARD=1    required to target card M or card A (half of the serving pair); loud warning
#
#   qual_card_select      sets QUAL_CARD, QUAL_BYID, QUAL_TAG, QUAL_SERVING; refuses a serving card
#                         without the override. Touches no device: dry runs call it too.
#   qual_card_resolve     sets QUAL_NODE (readlink -f, now) and QUAL_PCI; refuses a missing board and
#                         treats a board id that resolves to a serving card's node as that serving card.
#   qual_refuse_holders   refuses while a container or a host process can reach the target.
#   qual_card_recheck     readlink -f again right before the container starts; refuses if the node moved.
#   qual_reset_hint       the recovery lines after a hang, on stdout; it never resets anything itself.
QUAL_CARD_B=blackhole-F36F768B9A5CAFA0
QUAL_SERVING_CARDS='blackhole-CEF5729692C19E6D blackhole-3707293C249A5E67'
QUAL_TT_ROOT=/dev/tenstorrent
QUAL_BYID_ROOT=$QUAL_TT_ROOT/by-id
QUAL_SYS_ROOT=/sys

qual_card_label() {
  case ${1:-$QUAL_CARD} in
    blackhole-CEF5729692C19E6D) echo 'card M, half of the serving pair' ;;
    blackhole-3707293C249A5E67) echo 'card A, half of the serving pair' ;;
    "$QUAL_CARD_B") echo 'card B, the qualification card' ;;
    *) echo 'a board this harness does not name' ;;
  esac
}

qual_card_select() {
  QUAL_CARD=${QUAL_CARD:-$QUAL_CARD_B}
  case $QUAL_CARD in
    .*|*/*|*[!A-Za-z0-9._-]*)
      echo "refusing: QUAL_CARD=$QUAL_CARD is not a board id under $QUAL_BYID_ROOT (default $QUAL_CARD_B, card B)" >&2
      exit 1 ;;
  esac
  QUAL_BYID=$QUAL_BYID_ROOT/$QUAL_CARD
  case $QUAL_CARD in
    blackhole-CEF5729692C19E6D) QUAL_TAG=card-m ;;
    blackhole-3707293C249A5E67) QUAL_TAG=card-a ;;
    "$QUAL_CARD_B") QUAL_TAG=card-b ;;
    *) QUAL_TAG=$QUAL_CARD ;;
  esac
  QUAL_SERVING=0
  case " $QUAL_SERVING_CARDS " in
    *" $QUAL_CARD "*) qual_serving_override ;;
  esac
}

qual_serving_override() {
  QUAL_SERVING=1
  if [ "${ALLOW_SERVING_CARD:-0}" != 1 ]; then
    echo "refusing: QUAL_CARD=$QUAL_CARD is $(qual_card_label), which the CI gate and the endpoint use." >&2
    echo "  Qualify on card B (unset QUAL_CARD, or QUAL_CARD=$QUAL_CARD_B); ALLOW_SERVING_CARD=1 overrides." >&2
    exit 1
  fi
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' >&2
  echo "!!! WARNING: ALLOW_SERVING_CARD=1: this run is on $QUAL_CARD, $(qual_card_label)." >&2
  echo '!!! A CI gate that starts meanwhile fails its holder check, or resets the pair under this run before' >&2
  echo '!!! it opens the card; a hang here keeps the pair down until card M and card A are reset together.' >&2
  echo '!!! Check gh run list for the qwen-two-p150a-exclusive group before and after.' >&2
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' >&2
}

qual_is_char() {
  test -c "$1"
}

qual_card_resolve() {
  local card node
  QUAL_NODE=$(readlink -f -- "$QUAL_BYID" 2>/dev/null || true)
  if [ -z "$QUAL_NODE" ] || ! qual_is_char "$QUAL_NODE"; then
    echo "refusing: $QUAL_CARD ($(qual_card_label)) has no device node here: $QUAL_BYID does not resolve to one" >&2
    exit 1
  fi
  if [ "$QUAL_SERVING" != 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      node=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      if [ "$node" = "$QUAL_NODE" ]; then
        echo "### $QUAL_CARD resolves to $node, the node of $card ($(qual_card_label "$card"))" >&2
        qual_serving_override
      fi
    done
  fi
  QUAL_PCI=$(qual_pci_of "$QUAL_NODE")
  echo "### target card: $QUAL_CARD ($(qual_card_label)) -> $QUAL_NODE, PCI ${QUAL_PCI:-unknown}"
  if [ "$QUAL_SERVING" != 1 ]; then
    echo "### note: only the m3native gate spares this card; qwen-card-reset.yml and most other hardware workflows"
    echo "###   still act on every node - check gh run list for the qwen-two-p150a-exclusive group"
  fi
}

# A path's device numbers (major:minor, hex, following symlinks), or nothing.
qual_majmin_of() {
  stat -L -c '%t:%T' -- "$1" 2>/dev/null || true
}

# A device node's PCI address (0000:f4:00.0) from sysfs, or nothing.
qual_pci_of() {
  local majmin dev
  majmin=$(qual_majmin_of "${1:-}")
  case $majmin in
    *[!0-9a-f:]*|:*|*:|*:*:*) return 0 ;;
    *:*) ;;
    *) return 0 ;;
  esac
  dev=$(readlink -e -- "$QUAL_SYS_ROOT/dev/char/$((16#${majmin%%:*})):$((16#${majmin##*:}))/device" 2>/dev/null) || return 0
  case ${dev##*/} in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7]) echo "${dev##*/}" ;;
  esac
}

# Why a container (qual_refuse_holders' docker inspect lines) can reach the target; nothing when it cannot.
# A non-privileged container reaches a device only through its device list, its device cgroup rules and
# its device requests: a bind mount of /dev or /dev/tenstorrent shows node files it cannot open (every CI
# gate arm mounts /dev/tenstorrent read-only for its board-mapping check and is given card M and card A
# only). So for a non-serving target it names: --privileged; any device cgroup rule; a device request
# naming Tenstorrent (CDI); a mapped device path that is the target's node, has the target's device
# numbers, is a directory holding the node, or is a /dev/tenstorrent path that is not a device node now
# (which board it was given is unknowable); a mount of the target's node itself. For a serving target
# (ALLOW_SERVING_CARD=1) it names any container that can reach any Tenstorrent device: --privileged, a
# device cgroup rule, a Tenstorrent device request, or /dev or a tenstorrent path among its devices or
# mounts.
qual_container_reach() {
  local head=${1%%$'\n'*} line kind path resolved
  case $head in
    *' true') echo 'is --privileged (it can open every device)'; return 0 ;;
  esac
  while IFS= read -r line; do
    kind=${line%% *}
    path=${line#* }
    case $kind in
      rule) echo "has a device cgroup rule ($path)"; return 0 ;;
      req)
        case $path in
          *[Tt]enstorrent*) echo "has a device request for a Tenstorrent device ($path)"; return 0 ;;
        esac
        continue ;;
      dev|mnt) ;;
      *) continue ;;
    esac
    resolved=$(readlink -f -- "$path" 2>/dev/null || true)
    if [ "$QUAL_SERVING" = 1 ]; then
      case "$path $resolved" in
        "${QUAL_TT_ROOT%/*} "*|"${QUAL_TT_ROOT%/*}/ "*|*" ${QUAL_TT_ROOT%/*}"|*tenstorrent*)
          echo "has $path among its ${kind}s (a Tenstorrent device; the target is a serving card)"; return 0 ;;
      esac
      continue
    fi
    if [ "$resolved" = "$QUAL_NODE" ]; then
      echo "has $path among its ${kind}s, which is $QUAL_NODE (the target)"; return 0
    fi
    [ "$kind" = dev ] || continue
    if [ -n "${QUAL_MAJMIN:-}" ] && [ "$(qual_majmin_of "$path")" = "$QUAL_MAJMIN" ]; then
      echo "is given $path, a device with the target's numbers ($QUAL_MAJMIN)"; return 0
    fi
    case $QUAL_NODE in
      "$resolved"/*) echo "is given $path, a directory holding the target's node"; return 0 ;;
    esac
    case $path in
      "$QUAL_TT_ROOT"|"$QUAL_TT_ROOT"/*)
        if [ -z "$resolved" ] || ! qual_is_char "$resolved"; then
          echo "is given $path, which is not a device node now (it may be the target)"; return 0
        fi ;;
    esac
  done <<< "${1#*$'\n'}"
}

# Refuses while anything else can reach the target. Containers: qual_container_reach. Host processes:
# fuser on the target's node (with sudo -n when that works, else this user's processes, said so) - which
# also sees a process in a container holding it, however the container was given it - up to five tries
# two seconds apart (the rig's telemetry exporter holds every card for a moment every 30 s), refusing
# while any holder persists.
qual_refuse_holders() {
  local id info why st out try scope
  local pre=()
  QUAL_MAJMIN=$(qual_majmin_of "$QUAL_NODE")
  for id in $(docker ps -q); do
    info=$(docker inspect "$id" --format '{{.Name}} {{.HostConfig.Privileged}}{{println}}{{range .HostConfig.Devices}}dev {{println .PathOnHost}}{{end}}{{range .HostConfig.DeviceCgroupRules}}rule {{println .}}{{end}}{{range .HostConfig.DeviceRequests}}req {{.Driver}} {{println .DeviceIDs}}{{end}}{{range .Mounts}}mnt {{println .Source}}{{end}}') || continue
    why=$(qual_container_reach "$info")
    if [ -n "$why" ]; then
      echo "refusing: container ${info%% *} $why" >&2
      exit 1
    fi
  done
  echo "### containers: none can reach $QUAL_NODE ($QUAL_CARD)"
  if ! command -v fuser >/dev/null 2>&1; then
    echo "WARN: fuser is not installed; host processes holding $QUAL_NODE were not checked" >&2
    return 0
  fi
  scope="this user's processes only (no passwordless sudo)"
  if [ "$(id -u)" = 0 ]; then
    scope=all
  elif sudo -n true >/dev/null 2>&1; then
    pre=(sudo -n)
    scope=all
  fi
  for try in 1 2 3 4 5; do
    st=0
    out=$(${pre[@]+"${pre[@]}"} fuser -v "$QUAL_NODE" 2>&1) || st=$?
    if [ "$st" != 0 ] && [ -z "$out" ]; then
      echo "### device holders on $QUAL_NODE: none ($scope)"
      return 0
    fi
    [ "$try" = 5 ] || sleep 2
  done
  echo "refusing: host processes hold $QUAL_NODE (or fuser failed):" >&2
  echo "$out" >&2
  exit 1
}

# readlink -f again right before the container starts: refuses when a board is not on the node the holder
# check cleared (the boards re-enumerated in between - a switch event, or the gate resetting card A on the
# switch card B shares - so the old node may now be another board's). Args: [board id] [node]; the
# target by default.
qual_card_recheck() {
  local card=${1:-$QUAL_CARD} was=${2:-$QUAL_NODE} now
  now=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
  if [ -z "$now" ] || [ "$now" != "$was" ] || ! qual_is_char "$now"; then
    echo "refusing: $card ($(qual_card_label "$card")) was $was at the holder check and is ${now:-gone} now" >&2
    echo "  (the boards re-enumerated); nothing was launched - check the boards, then run again" >&2
    exit 1
  fi
  echo "### $card is still $now"
}

# The recovery lines after a hang. The node and PCI address are looked up again now (nodes renumber),
# and the reset command resolves the board id again when it is run: tt-smi -r takes a /dev/tenstorrent
# path, while a bare number there is tt-smi's own board index, which renumbers too - so none is printed.
# The command runs tt-smi only when readlink -e resolved every board id (an empty argument would make
# tt-smi -r reset every board). The PCI address is what identifies the board's row in tt-smi -ls.
qual_reset_hint() {
  local node pci card n p var vars='m a' cmd= args=
  node=$(readlink -f -- "$QUAL_BYID" 2>/dev/null || true)
  pci=$(qual_pci_of "$node")
  echo "HANG RECOVERY for $QUAL_CARD ($(qual_card_label)), once the container is gone; nothing here resets a card."
  echo "  It is ${node:-absent} now, PCI ${pci:-unknown}; nodes renumber, so the reset below resolves the board id"
  echo "  when it is run. Never a bare number: tt-smi -r reads one as tt-smi's own board index, which renumbers too."
  if [ "$QUAL_SERVING" = 1 ]; then
    for card in $QUAL_SERVING_CARDS; do
      n=$(readlink -f -- "$QUAL_BYID_ROOT/$card" 2>/dev/null || true)
      p=$(qual_pci_of "$n")
      echo "  $card ($(qual_card_label "$card")) is ${n:-absent} now, PCI ${p:-unknown}."
      var=${vars%% *}
      vars=${vars#* }
      cmd="$cmd$var=\$(readlink -e $QUAL_BYID_ROOT/$card) && "
      args="$args \"\$$var\""
    done
    echo "  It is half of the serving pair: one link end reset alone leaves the mesh at 1x1. Check gh run list for the"
    echo "  qwen-two-p150a-exclusive group, confirm both PCI addresses in ~/.local/bin/tt-smi -ls, then reset card M and"
    echo "  card A TOGETHER, in one call:"
    echo "    $cmd~/.local/bin/tt-smi -r$args"
    echo "  then a passing smoke run."
  else
    if [ -n "$pci" ]; then
      echo "  CONFIRM its row (PCI BDF $pci): ~/.local/bin/tt-smi -ls | grep -i '${pci#0000:}'; then reset it alone:"
    else
      echo "  CONFIRM in ~/.local/bin/tt-smi -ls which row is ${node:-this board} (its PCI address is unknown here); then reset it alone:"
    fi
    echo "    n=\$(readlink -e $QUAL_BYID) && ~/.local/bin/tt-smi -r \"\$n\""
    echo "  then a passing smoke run. Never card M or card A: they are the serving pair, and this card needs neither."
  fi
}
# <<< qual_card.sh
qual_card_select
DRY=${K64J_BUILD_DRY_RUN:-0}
case $DRY in
  0|1) ;;
  *) echo "refusing: K64J_BUILD_DRY_RUN=$DRY (0 or 1)" >&2; exit 1 ;;
esac
if [ -n "${CARD_B_ARGS:-}" ]; then
  echo "refusing: CARD_B_ARGS='$CARD_B_ARGS': build_k64j.sh takes no arguments from the job file; set K64J_* in CARD_B_ENV" >&2
  exit 1
fi
if [ $# -gt 1 ]; then
  echo "refusing: build_k64j.sh takes at most one argument, the graft directory (got $#)" >&2
  exit 1
fi

S=$(cd "$(dirname "$0")" && pwd)
CHECKOUT=$(cd "$S/../../.." && pwd)
for dir in sdpa_decode_slice sdpa_decode_qwen sdpa_prefill_chain; do
  test -d "$S/../$dir" || { echo "refusing: $S/../$dir missing (the checkout carries k64j and its three siblings)" >&2; exit 1; }
done
SL=$(cd "$S/../sdpa_decode_slice" && pwd)
DS=$(cd "$S/../sdpa_decode_qwen" && pwd)
PS=$(cd "$S/../sdpa_prefill_chain" && pwd)
KS=$DS/stage3
KJ=$S/kernels
DBASE=${K64J_DECODE_BASE_DIR:-$HOME/kwork64/k64f}
PBASE=${K64J_PF_BASE_DIR:-$HOME/kwork64/k64g}
BASE=${K64J_BASE_GRAFT:-$HOME/opgraft-K64i}
BASE_SHA=${K64J_BASE_SHA256-$K64I_TTNNCPP}
GRAFT=${1:-${K64J_GRAFT:-$HOME/opgraft-K64j}}
G=$GRAFT.partial
WROOT=${K64J_WORK_ROOT:-$HOME/kwork64/k64j}
IMAGES=${K64J_IMAGES:-$GATE_IMAGE}
IMAGES=${IMAGES//,/ }
OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
XOPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer
D=$OPS/sdpa_decode
F=$D/device/sdpa_decode_program_factory.cpp
KD=$D/device/kernels
P=$OPS/sdpa
PF=$P/device/sdpa_program_factory.cpp
PK=$P/device/kernels
TB_SO=/opt/tt-metal/build_Release/ttnn/_ttnncpp.so
SAVED_DECODE=$DBASE/sdpa_decode_program_factory.cpp.$FACTORY_BASE
SAVED_PF=$PBASE/sdpa_program_factory.cpp.$PF_BASE

# The graft is never the base, never inside the checkout (actions/checkout cleans the workspace), and neither is the
# work root.
canon() {  # one absolute spelling of a path (under MSYS, C:/x, /c/x and /tmp/x may name the same place)
  local path=$1
  if command -v cygpath >/dev/null 2>&1; then
    path=$(cygpath -u -- "$path")
  fi
  readlink -m -- "$path"
}
under() {  # path root -> true when path is root or inside it
  local path root
  path=$(canon "$1")
  root=$(canon "$2")
  case "$path/" in "$root/"*) return 0 ;; esac
  return 1
}
case $GRAFT in
  ''|/|.|..) echo "refusing: graft directory '$GRAFT'" >&2; exit 1 ;;
esac
if under "$GRAFT" "$BASE" || under "$BASE" "$GRAFT"; then
  echo "refusing: the graft $GRAFT and the base $BASE overlap (the base is K64i, which the build copies)" >&2
  exit 1
fi
for path in "$GRAFT" "$WROOT"; do
  if under "$path" "$CHECKOUT"; then
    echo "refusing: $path is inside the checkout $CHECKOUT (the next checkout cleans it)" >&2
    exit 1
  fi
done

stamp=$(date +%Y%m%dT%H%M%S)
# Commands that change something or talk to docker go through x: printed and not run in a dry run. Their printed
# form goes to fd 3, the console, so $(x ...) captures nothing in a dry run.
exec 3>&1
x() {
  if [ "$DRY" = 1 ]; then
    echo "### run: $(printf '%q ' "$@")" >&3
    return 0
  fi
  "$@"
}
fail() {
  echo "FAIL: $*" >&2
  exit 1
}
need() {  # label actual expected (an empty actual in a dry run is a command that was not run)
  if [ "$DRY" = 1 ] && [ -z "$2" ]; then
    echo "dry   $1: must be ${3:0:16}"
    return 0
  fi
  if [ "$2" != "$3" ]; then
    echo "FAIL: $1 is ${2:-missing}, expected $3" >&2
    exit 1
  fi
  echo "ok    $1 ${2:0:16}"
}
hsha() { sha256sum "$1" 2>/dev/null | cut -c1-64 || true; }
csha() { x docker exec ttbuild sha256sum "$1" | cut -c1-64; }
cexists() { x docker exec ttbuild test -e "$1"; }
count() {  # file fixed-string -> number of matching strings (0 is not an error here)
  strings "$1" | grep -cF -- "$2" || true
}
qwen_strings() {  # every distinct QWEN_ / [QWEN- string of a binary, sorted
  strings "$1" | grep -E 'QWEN_|\[QWEN-' | sort -u || true
}

# Where this runs (the card-B runner's user and HOME are what the defaults above hang on) and whether every path the
# build writes is writable, with its free space. A dry run only reports; a real build stops here on an unwritable one.
nearest() {  # the nearest existing ancestor of a path
  local path
  path=$(canon "$1")
  while [ ! -e "$path" ] && [ "$path" != / ]; do
    path=$(dirname "$path")
  done
  echo "$path"
}
echo "### host $(hostname 2>/dev/null || echo unknown) user $(id -un 2>/dev/null || echo unknown) HOME $HOME"
for path in "$GRAFT" "$WROOT" "$DBASE" "$PBASE"; do
  at=$(nearest "$path")
  free=$(df -Pk "$at" 2>/dev/null | awk 'NR == 2 { printf "%d MB free", $4 / 1024 }' || true)
  if [ -w "$at" ]; then
    echo "### $path: its nearest existing directory $at is writable, ${free:-free space unknown}"
  elif [ "$DRY" = 1 ]; then
    echo "### $path: its nearest existing directory $at is NOT WRITABLE by $(id -un 2>/dev/null) (the build would stop here)"
  else
    fail "$path: its nearest existing directory $at is not writable by $(id -un 2>/dev/null)"
  fi
done

if [ "$DRY" = 1 ]; then
  W=$(mktemp -d "${TMPDIR:-/tmp}/k64j-dry.XXXXXX")   # the read-only checks' scratch, removed on exit
  echo "### build_k64j DRY RUN $(date -Is) graft=$GRAFT base=$BASE card context $QUAL_CARD ($QUAL_TAG): nothing is built"
  x mkdir -p "$WROOT/work-$stamp/backup"
else
  command -v docker >/dev/null 2>&1 || fail "docker is not on PATH"
  command -v strings >/dev/null 2>&1 || fail "strings (binutils) is not on PATH: step 6 needs it"
  W=$WROOT/work-$stamp
  mkdir -p "$W/backup"
  echo "### build_k64j $(date -Is) work=$W graft=$GRAFT base=$BASE card context $QUAL_CARD ($QUAL_TAG): no device is opened"
fi

# The EXIT trap: ttbuild's sources back (from the first staging copy on), and the dry run's scratch removed.
armed=0
restored=0
built=0
rebuilt=0
dcur=
pcur=
restore_ttbuild() {
  if [ "$armed" = "0" ] || [ "$restored" = "1" ] || [ "${K64J_KEEP_STAGED:-}" = "1" ]; then
    return 0
  fi
  x docker cp "$SAVED_DECODE" "ttbuild:$F"
  x docker cp "$SAVED_PF" "ttbuild:$PF"
  for kernel in "${DECODE_KERNELS[@]}"; do
    if [ -e "$W/backup/$(basename "$kernel").replaced" ]; then
      x docker cp "$W/backup/$(basename "$kernel").replaced" "ttbuild:$KD/$kernel"
    else
      x docker exec ttbuild rm -f "$KD/$kernel"
    fi
  done
  if [ -e "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" ]; then
    x docker cp "$W/backup/reader_interleaved_qwen_chain.cpp.replaced" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  else
    x docker exec ttbuild rm -f "$PK/dataflow/reader_interleaved_qwen_chain.cpp"
  fi
  # The restored files carry the saved copies' mtimes, older than the objects ninja just built from the qwen
  # factories; touch them so ninja recompiles the unity TUs that include them.
  x docker exec ttbuild touch "$F" "$PF"
  restored=1
  if [ "$DRY" = 1 ]; then
    echo "ttbuild restore: the commands above (factories back from the saved bases, qwen kernel files removed or put back)"
  else
    echo "ttbuild restored: decode factory $(csha $F | cut -c1-16), prefill factory $(csha $PF | cut -c1-16) (touched), qwen kernel files removed or put back"
  fi
}
on_exit() {
  local status=$?
  restore_ttbuild || echo "WARN: ttbuild restore failed; check $F and $PF in ttbuild" >&2
  if [ "$DRY" != 1 ] && [ "$built" = "1" ] && [ "$rebuilt" = "0" ] && [ "${K64J_KEEP_STAGED:-}" != "1" ]; then
    echo "WARN  ttbuild build_Release still holds the qwen objects; the next ninja run recompiles the touched factories, but do not copy a .so out of build_Release before one" >&2
  fi
  if [ "$DRY" = 1 ]; then
    rm -rf "$W"
  elif [ "$status" != 0 ] && [ -e "$G" ]; then
    echo "note  $G is left for inspection; the next run removes it" >&2
  fi
}
trap on_exit EXIT

# ---------- 2. preflight: the checkout ----------
for path in "$KJ/dataflow/reader_decode_qwen.cpp" "$KJ/dataflow/reader_decode_qwen_slice.cpp" \
            "$KJ/compute/sdpa_flash_decode_qwen.cpp" "$KJ/dataflow/writer_decode_qwen_slice.cpp" \
            "$S/apply_factory_k64j.py" "$S/make_k64j_kernels.py" "$SL/apply_factory_slice.py" "$SL/make_slice_kernels.py" \
            "$SL/reader_decode_qwen_slice.cpp" "$SL/writer_decode_qwen_slice.cpp" "$DS/apply_factory_qwen.py" \
            "$DS/make_qwen_kernels.py" "$KS/reader_decode_qwen.cpp" "$KS/sdpa_flash_decode_qwen.cpp" \
            "$PS/apply_factory_pf.py" "$PS/make_pf_reader.py" "$PS/reader_interleaved_qwen_chain.cpp"; do
  test -s "$path" || fail "$path missing"
  if grep -q $'\r' "$path"; then fail "$path has CRLF line endings"; fi
done
for kernel in "${DECODE_KERNELS[@]}"; do
  need "shipped K64j $kernel" "$(hsha "$KJ/$kernel")" "$(k64j_sha "$kernel")"
done
need "shipped K64i stage-3 reader" "$(hsha "$KS/reader_decode_qwen.cpp")" $READER_QWEN
need "shipped K64i qwen compute" "$(hsha "$KS/sdpa_flash_decode_qwen.cpp")" $COMPUTE_QWEN
need "shipped K64i slice reader" "$(hsha "$SL/reader_decode_qwen_slice.cpp")" $READER_SLICE
need "shipped K64i slice writer" "$(hsha "$SL/writer_decode_qwen_slice.cpp")" $WRITER_SLICE
need "shipped reader_interleaved_qwen_chain.cpp" "$(hsha "$PS/reader_interleaved_qwen_chain.cpp")" $PF_READER
# The committed K64j kernels are the committed K64i kernels plus R10 / C3 / W4 (read-only; writes nothing).
python3 -B "$S/make_k64j_kernels.py" --check "$KJ"
if [ "${K64J_SKIP_IMAGE_COMPARE:-}" != "1" ]; then
  case " $IMAGES " in
    *" $GATE_IMAGE "*) ;;
    *) fail "K64J_IMAGES must include the card-B / C2 arm image $GATE_IMAGE" ;;
  esac
fi

# ---------- 2. preflight: the base graft (K64i; read-only, also in a dry run when it exists here) ----------
base_so=
if [ "$DRY" = 1 ] && [ ! -e "$BASE" ]; then
  echo "### dry run: $BASE does not exist here; the base graft was not checked"
else
  for part in _ttnncpp.so _ttnn.so attn_prep nlp_concat_heads_decode sdpa_decode sdpa MANIFEST.sha256; do
    test -e "$BASE/$part" || fail "$BASE/$part missing (the base is K64i, build_k64i.sh)"
  done
  (cd "$BASE" && sha256sum -c --quiet MANIFEST.sha256) || fail "$BASE/MANIFEST.sha256 does not verify"
  echo "ok    $BASE/MANIFEST.sha256 verifies"
  base_so=$(hsha "$BASE/_ttnncpp.so")
  if [ -n "$BASE_SHA" ]; then
    need "base graft _ttnncpp.so (K64i)" "$base_so" "$BASE_SHA"
  fi
  BK=$BASE/sdpa_decode/device/kernels
  for kernel in "${DECODE_KERNELS[@]}"; do
    need "base graft $kernel (K64i)" "$(hsha "$BK/$kernel")" "$(k64i_sha "$kernel")"
  done
  need "base graft reader_decode_all.cpp" "$(hsha "$BK/dataflow/reader_decode_all.cpp")" $READER_ALL
  need "base graft writer_decode_all.cpp" "$(hsha "$BK/dataflow/writer_decode_all.cpp")" $WRITER_ALL
  need "base graft sdpa_flash_decode.cpp" "$(hsha "$BK/compute/sdpa_flash_decode.cpp")" $COMPUTE_ALL
  need "base graft dataflow_common.hpp" "$(hsha "$BK/dataflow/dataflow_common.hpp")" $DATAFLOW_COMMON
  need "base graft rt_args_common.hpp" "$(hsha "$BK/rt_args_common.hpp")" $RT_ARGS_COMMON
  need "base graft sdpa_decode factory (on disk, audited)" "$(hsha "$BASE/sdpa_decode/device/sdpa_decode_program_factory.cpp")" $FACTORY_BASE
  need "base graft reader_interleaved_qwen_chain.cpp" "$(hsha "$BASE/sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp")" $PF_READER
  if command -v strings >/dev/null 2>&1; then
    test "$(count "$BASE/_ttnncpp.so" QWEN_SDPA_TREE_SCRATCH_ROUNDS)" -ge 1 || fail "$BASE/_ttnncpp.so lacks the tree-scratch factory"
    test "$(count "$BASE/_ttnncpp.so" "$SHARE_MARKER")" -ge 1 || fail "$BASE/_ttnncpp.so lacks '$SHARE_MARKER' (not K64f onward)"
    test "$(count "$BASE/_ttnncpp.so" "$SLICE_MARKER")" -ge 1 || fail "$BASE/_ttnncpp.so lacks '$SLICE_MARKER' (not K64i)"
    test "$(count "$BASE/_ttnncpp.so" '[QWEN-SDPA-PF] flags=')" -ge 1 || fail "$BASE/_ttnncpp.so lacks [QWEN-SDPA-PF] (not K64g onward)"
    test "$(count "$BASE/_ttnncpp.so" "$EXTENT_MARKER")" -eq 0 || fail "$BASE/_ttnncpp.so already has the K64j factory"
    qwen_strings "$BASE/_ttnncpp.so" > "$W/base-qwen-strings.txt"
    echo "ok    base graft ${base_so:0:16}: $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings recorded"
  else
    echo "### dry run: no strings here; the base graft's QWEN strings were not read"
  fi
fi

# ---------- 2. preflight: the saved factory bases (read-only; a dry run patches into its scratch) ----------
for pair in "decode|$SAVED_DECODE|$FACTORY_BASE" "prefill|$SAVED_PF|$PF_BASE"; do
  label=${pair%%|*}
  rest=${pair#*|}
  path=${rest%|*}
  if [ -e "$path" ]; then
    need "saved $label base factory" "$(hsha "$path")" "${rest##*|}"
  else
    echo "note  no saved $label base at $path yet: step 3 saves it from ttbuild when ttbuild holds the base, else fails"
  fi
done
if [ "$DRY" = 1 ]; then
  if [ -e "$SAVED_DECODE" ]; then
    python3 -B "$S/apply_factory_k64j.py" "$SAVED_DECODE" --out "$W/k64j-factory.cpp" >/dev/null
    need "the saved base patched to K64j (dry-run scratch)" "$(hsha "$W/k64j-factory.cpp")" $FACTORY_K64J
  fi
  if [ -e "$SAVED_PF" ]; then
    python3 -B "$PS/apply_factory_pf.py" "$SAVED_PF" --out "$W/pf-factory.cpp" >/dev/null
    need "the saved prefill base patched (dry-run scratch)" "$(hsha "$W/pf-factory.cpp")" $PF_FACTORY
  fi
fi

# ---------- 2. preflight: ttbuild and the images ----------
if [ "$DRY" = 1 ]; then
  x docker ps --format '{{.Names}}'
else
  docker ps --format '{{.Names}}' | grep -qx ttbuild || fail "ttbuild container is not running"
  echo "ok    ttbuild is running"
fi
dcur=$(csha $F)
if [ "$DRY" != 1 ]; then
  case "$dcur" in
    "$FACTORY_BASE") ;;
    "$FACTORY_QWEN_STAGE1"|"$FACTORY_QWEN_STAGE3"|"$FACTORY_QWEN_STAGE4"|"$FACTORY_K64J")
      echo "note  ttbuild decode factory is a qwen factory (${dcur:0:8}, a *_KEEP_STAGED run); step 3 patches from the saved 3e0a69af" ;;
    "$FACTORY_UNPATCHED") fail "ttbuild decode factory is the unpatched 05708e6d; run the K64d build (tree-scratch) first" ;;
    *) fail "unexpected ttbuild decode factory ${dcur:-missing}" ;;
  esac
  echo "ok    ttbuild decode factory ${dcur:0:16}"
fi
pcur=$(csha $PF)
if [ "$DRY" != 1 ]; then
  case "$pcur" in
    "$PF_BASE") ;;
    "$PF_FACTORY") echo "note  ttbuild prefill factory is [QWEN-SDPA-PF]-patched (a *_KEEP_STAGED run); step 3 patches from the saved fd8c0676" ;;
    *) fail "unexpected ttbuild prefill factory ${pcur:-missing}" ;;
  esac
  echo "ok    ttbuild prefill factory ${pcur:0:16}"
fi
need "ttbuild reader_decode_all.cpp" "$(csha $KD/dataflow/reader_decode_all.cpp)" $READER_ALL
need "ttbuild sdpa_flash_decode.cpp" "$(csha $KD/compute/sdpa_flash_decode.cpp)" $COMPUTE_ALL
need "ttbuild writer_decode_all.cpp" "$(csha $KD/dataflow/writer_decode_all.cpp)" $WRITER_ALL
need "ttbuild decode dataflow_common.hpp" "$(csha $KD/dataflow/dataflow_common.hpp)" $DATAFLOW_COMMON
need "ttbuild rt_args_common.hpp" "$(csha $KD/rt_args_common.hpp)" $RT_ARGS_COMMON
need "ttbuild reader_interleaved.cpp" "$(csha $PK/dataflow/reader_interleaved.cpp)" $PF_READER_BASE
# The committed kernels must regenerate from ttbuild's own originals, not just from the committed K64i kernels.
x docker cp ttbuild:$KD/dataflow/reader_decode_all.cpp "$W/reader_decode_all.cpp"
x docker cp ttbuild:$KD/compute/sdpa_flash_decode.cpp "$W/sdpa_flash_decode.cpp"
x docker cp ttbuild:$KD/dataflow/writer_decode_all.cpp "$W/writer_decode_all.cpp"
x docker cp ttbuild:$PK/dataflow/reader_interleaved.cpp "$W/reader_interleaved.cpp"
x python3 -B "$S/make_k64j_kernels.py" --reader "$W/reader_decode_all.cpp" --writer "$W/writer_decode_all.cpp" \
  --compute "$W/sdpa_flash_decode.cpp" --check "$KJ"
x python3 -B "$PS/make_pf_reader.py" --reader "$W/reader_interleaved.cpp" --check "$PS"
# attn_prep / nlp_concat_heads_decode in ttbuild must be what the base graft's .so was built from.
x rm -rf "$W/tb-attn_prep" "$W/tb-nlp_concat_heads_decode"
x docker cp ttbuild:$OPS/attn_prep "$W/tb-attn_prep"
x docker cp ttbuild:$XOPS/nlp_concat_heads_decode "$W/tb-nlp_concat_heads_decode"
if [ "$DRY" != 1 ]; then
  diff -r "$W/tb-attn_prep" "$BASE/attn_prep" >/dev/null || fail "ttbuild attn_prep differs from $BASE/attn_prep"
  diff -r "$W/tb-nlp_concat_heads_decode" "$BASE/nlp_concat_heads_decode" >/dev/null \
    || fail "ttbuild nlp_concat_heads_decode differs from $BASE's"
  echo "ok    ttbuild attn_prep and nlp_concat_heads_decode equal $BASE's"
fi
if [ "${K64J_SKIP_IMAGE_COMPARE:-}" != "1" ]; then
  for image in $IMAGES; do
    if [ "$DRY" = 1 ]; then
      x docker image inspect "$image"
    else
      docker image inspect "$image" >/dev/null 2>&1 \
        || fail "image $image is not present; set K64J_IMAGES to the present images (keeping $GATE_IMAGE)"
    fi
  done
  [ "$DRY" = 1 ] || echo "ok    images present: $(for image in $IMAGES; do printf '%s ' "${image:7:12}"; done)"
fi

# ---------- 3. stage both factories and the five kernels (backing up anything replaced) ----------
# 3a. the saved bases (from ttbuild when it holds them), then the decode factory at K64j
x docker cp ttbuild:$F "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}"
x docker cp ttbuild:$PF "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}"
x mkdir -p "$DBASE" "$PBASE"
if [ "$dcur" = "$FACTORY_BASE" ] && [ ! -e "$SAVED_DECODE" ]; then
  x cp "$W/backup/sdpa_decode_program_factory.cpp.${dcur:0:8}" "$SAVED_DECODE"
fi
if [ "$pcur" = "$PF_BASE" ] && [ ! -e "$SAVED_PF" ]; then
  x cp "$W/backup/sdpa_program_factory.cpp.${pcur:0:8}" "$SAVED_PF"
fi
if [ "$DRY" != 1 ]; then
  test -s "$SAVED_DECODE" || fail "no saved 3e0a69af decode factory in $DBASE (and ttbuild does not hold it)"
  test -s "$SAVED_PF" || fail "no saved fd8c0676 prefill factory in $PBASE (and ttbuild does not hold it)"
fi
need "saved decode base factory" "$( [ "$DRY" = 1 ] || hsha "$SAVED_DECODE")" $FACTORY_BASE
need "saved prefill base factory" "$( [ "$DRY" = 1 ] || hsha "$SAVED_PF")" $PF_BASE
x python3 -B "$S/apply_factory_k64j.py" "$SAVED_DECODE" --out "$W/sdpa_decode_program_factory.cpp"
need "patched decode factory (K64j)" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_decode_program_factory.cpp")" $FACTORY_K64J
# 3b. the prefill factory, exactly as build_k64g.sh / build_k64i.sh
x python3 -B "$PS/apply_factory_pf.py" "$SAVED_PF" --out "$W/sdpa_program_factory.cpp"
need "patched prefill factory" "$( [ "$DRY" = 1 ] || hsha "$W/sdpa_program_factory.cpp")" $PF_FACTORY

# 3c. stage (from here on ttbuild is modified; the EXIT trap puts it back)
armed=1
x docker cp "$W/sdpa_decode_program_factory.cpp" "ttbuild:$F"
x docker cp "$W/sdpa_program_factory.cpp" "ttbuild:$PF"
x docker exec ttbuild touch "$F" "$PF"   # newer than every object, whatever mtime docker cp gave them
need "ttbuild decode factory now" "$(csha $F)" $FACTORY_K64J
need "ttbuild prefill factory now" "$(csha $PF)" $PF_FACTORY
for kernel in "${DECODE_KERNELS[@]}"; do
  if cexists "$KD/$kernel"; then
    x docker cp "ttbuild:$KD/$kernel" "$W/backup/$(basename "$kernel").replaced"
  fi
  x docker cp "$KJ/$kernel" "ttbuild:$KD/$kernel"
done
if cexists "$PK/dataflow/reader_interleaved_qwen_chain.cpp"; then
  x docker cp "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp" "$W/backup/reader_interleaved_qwen_chain.cpp.replaced"
fi
x docker cp "$PS/reader_interleaved_qwen_chain.cpp" "ttbuild:$PK/dataflow/reader_interleaved_qwen_chain.cpp"
for kernel in "${DECODE_KERNELS[@]}"; do
  need "ttbuild $kernel" "$(csha "$KD/$kernel")" "$(k64j_sha "$kernel")"
done
need "ttbuild reader_interleaved_qwen_chain.cpp" "$(csha $PK/dataflow/reader_interleaved_qwen_chain.cpp)" $PF_READER

# ---------- 4. build ----------
echo "### ninja start $(date -Is)"
built=1
x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
echo "### ninja done $(date -Is)"

# ---------- 5. assemble the graft: K64i plus the new .so files and the four K64j kernels ----------
x rm -rf "$G"
x mkdir -p "$G"
x cp -a "$BASE/." "$G/"
x rm -f "$G/MANIFEST.sha256"
x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so "$G/_ttnncpp.so"
x docker cp ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnn.so "$G/_ttnn.so"
for kernel in "${DECODE_KERNELS[@]}"; do
  x cp "$KJ/$kernel" "$G/sdpa_decode/device/kernels/$kernel"
done

# ---------- 7. restore ttbuild (before the verify, as build_k64i.sh does; the EXIT trap covers a failure) ----------
if [ "${K64J_KEEP_STAGED:-}" = "1" ]; then
  echo "note  K64J_KEEP_STAGED=1: ttbuild keeps the qwen factories and kernels"
else
  restore_ttbuild
  need "ttbuild decode factory restored" "$(csha $F)" $FACTORY_BASE
  need "ttbuild prefill factory restored" "$(csha $PF)" $PF_BASE
  echo "### ninja (restore build_Release to the audited factories) start $(date -Is)"
  x docker exec ttbuild bash -c "cd /opt/tt-metal && set -o pipefail && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so 2>&1 | tail -6"
  echo "### ninja (restore) done $(date -Is)"
  tb_flags=$(x docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA] flags=' $TB_SO || true")
  tb_extent=$(x docker exec ttbuild bash -c "grep -caF -- '$EXTENT_MARKER' $TB_SO || true")
  tb_pf=$(x docker exec ttbuild bash -c "grep -caF -- '[QWEN-SDPA-PF]' $TB_SO || true")
  tb_scratch=$(x docker exec ttbuild bash -c "grep -caF -- QWEN_SDPA_TREE_SCRATCH_ROUNDS $TB_SO || true")
  if [ "$DRY" != 1 ]; then
    test "$tb_flags" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA] branch; build_Release is dirty"
    test "$tb_extent" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the K64j runtime extent; build_Release is dirty"
    test "$tb_pf" = "0" || fail "ttbuild's rebuilt _ttnncpp.so still has the [QWEN-SDPA-PF] branch; build_Release is dirty"
    test "$tb_scratch" -ge 1 || fail "ttbuild's rebuilt _ttnncpp.so lost the tree-scratch factory"
    echo "ok    ttbuild build_Release rebuilt from the audited factories (no qwen branch, tree scratch present)"
  fi
  rebuilt=1
fi

# ---------- 6. verify ----------
echo "### verify $G"
if [ "$DRY" = 1 ]; then
  echo "### dry run: step 6 reads the assembled graft; its checks were not run"
else
  so=$G/_ttnncpp.so
  test "$(hsha "$so")" != "$base_so" || fail "the build produced the base graft's _ttnncpp.so unchanged"
  qwen_strings "$so" > "$W/graft-qwen-strings.txt"
  lost=$(comm -23 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt")
  if [ -n "$lost" ]; then
    echo "FAIL: QWEN strings of $BASE/_ttnncpp.so missing from the new .so (a graft .so replaces the whole binary):" >&2
    echo "$lost" >&2
    exit 1
  fi
  echo "ok    every one of the base's $(wc -l < "$W/base-qwen-strings.txt") distinct QWEN strings is in the new .so ($(wc -l < "$W/graft-qwen-strings.txt") now)"
  for marker in "${K64J_MARKERS[@]}" "${STAGE4_MARKERS[@]}" "${PF_MARKERS[@]}" '[QWEN-SDPA] flags=' QWEN_SDPA_TREE_SCRATCH_ROUNDS \
                reader_decode_qwen.cpp "$SHARE_MARKER"; do
    test "$(count "$so" "$marker")" -ge 1 || fail "the new _ttnncpp.so lacks '$marker'"
    echo "ok    strings: '$marker'"
  done
  test "$(count "$so" "$STAGE1_REFUSAL")" -eq 0 || fail "the new _ttnncpp.so refuses KV share (the stage-1 decode factory was linked)"
  combined=$(count "$so" qwen_draft_fp32_intermediates)
  test "$combined" = "$(count "$BASE/_ttnncpp.so" qwen_draft_fp32_intermediates)" \
    || fail "the combined prefill factory's qwen_draft_fp32_intermediates count moved from $BASE's"
  echo "ok    qwen_draft_fp32_intermediates count $combined, as $BASE"
  for op in attn_prep nlp_concat_heads_decode sdpa; do
    diff -r "$BASE/$op" "$G/$op" >/dev/null || fail "$G/$op differs from $BASE/$op"
  done
  echo "ok    attn_prep, nlp_concat_heads_decode and sdpa identical to $BASE"
  differences=$(diff -rq "$BASE/sdpa_decode" "$G/sdpa_decode" || true)
  expected=$(for kernel in "${DECODE_KERNELS[@]}"; do
    printf 'Files %s and %s differ\n' "$BASE/sdpa_decode/device/kernels/$kernel" "$G/sdpa_decode/device/kernels/$kernel"
  done)
  if [ "$(echo "$differences" | sort)" != "$(echo "$expected" | sort)" ]; then
    echo "FAIL: the graft's sdpa_decode differs from $BASE's by more than the four K64j kernels:" >&2
    echo "$differences" >&2
    exit 1
  fi
  echo "ok    sdpa_decode = $BASE's with the four K64j kernels in place of K64i's"
  GK=$G/sdpa_decode/device/kernels
  need "graft sdpa_decode factory (on disk, audited)" "$(hsha "$G/sdpa_decode/device/sdpa_decode_program_factory.cpp")" $FACTORY_BASE
  for kernel in "${DECODE_KERNELS[@]}"; do
    need "graft $kernel (K64j)" "$(hsha "$GK/$kernel")" "$(k64j_sha "$kernel")"
  done
  need "graft reader_decode_all.cpp" "$(hsha "$GK/dataflow/reader_decode_all.cpp")" $READER_ALL
  need "graft writer_decode_all.cpp" "$(hsha "$GK/dataflow/writer_decode_all.cpp")" $WRITER_ALL
  need "graft sdpa_flash_decode.cpp" "$(hsha "$GK/compute/sdpa_flash_decode.cpp")" $COMPUTE_ALL
  need "graft dataflow_common.hpp" "$(hsha "$GK/dataflow/dataflow_common.hpp")" $DATAFLOW_COMMON
  need "graft rt_args_common.hpp" "$(hsha "$GK/rt_args_common.hpp")" $RT_ARGS_COMMON
fi
# Mounting the graft's op directories must change nothing in the container but the qwen kernels, in every image it
# will be mounted into.
if [ "${K64J_SKIP_IMAGE_COMPARE:-}" = "1" ]; then
  echo "WARN  K64J_SKIP_IMAGE_COMPARE=1: sdpa_decode/, sdpa/ and the image binaries were NOT compared against $IMAGES"
else
  for image in $IMAGES; do
    cid=$(x docker create --network none --entrypoint true "$image")
    [ "$DRY" = 1 ] && cid='<container>'
    x rm -rf "$W/image-sdpa" "$W/image-sdpa_decode" "$W/image-ttnncpp.so"
    x docker cp "$cid:$P" "$W/image-sdpa"
    x docker cp "$cid:$D" "$W/image-sdpa_decode"
    x docker cp -L "$cid:/opt/tt-metal/build_Release/lib/_ttnncpp.so" "$W/image-ttnncpp.so"
    x docker rm "$cid" >/dev/null
    if [ "$DRY" = 1 ]; then
      continue
    fi
    qwen_strings "$W/image-ttnncpp.so" > "$W/image-${image:7:12}-qwen-strings.txt"
    image_lost=$(comm -23 "$W/image-${image:7:12}-qwen-strings.txt" "$W/graft-qwen-strings.txt")
    if [ -n "$image_lost" ]; then
      echo "FAIL: QWEN strings of image ${image:7:12}'s _ttnncpp.so missing from the new .so:" >&2
      echo "$image_lost" >&2
      exit 1
    fi
    echo "ok    every one of image ${image:7:12}'s $(wc -l < "$W/image-${image:7:12}-qwen-strings.txt") distinct QWEN strings is in the new .so"
    rm -f "$W/image-ttnncpp.so"
    # Only qwen-named kernel files may differ: added by the graft, or the image's own (older) qwen kernels.
    bad=
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      case $line in
        "Only in $G/sdpa_decode/device/kernels/dataflow: "*qwen*.cpp|"Only in $G/sdpa_decode/device/kernels/compute: "*qwen*.cpp) ;;
        "Files $W/image-sdpa_decode/device/kernels/"*qwen*".cpp and $G/sdpa_decode/device/kernels/"*qwen*".cpp differ") ;;
        *) bad="$bad$line"$'\n' ;;
      esac
    done <<< "$(diff -rq "$W/image-sdpa_decode" "$G/sdpa_decode" || true)"
    if [ -n "$bad" ]; then
      echo "FAIL: the graft's sdpa_decode differs from image ${image:7:12}'s by more than qwen kernels:" >&2
      printf '%s' "$bad" >&2
      exit 1
    fi
    echo "ok    sdpa_decode = image ${image:7:12}'s but for qwen kernels"
    differences=$(diff -rq "$W/image-sdpa" "$G/sdpa" || true)
    case $differences in
      ''|"Only in $G/sdpa/device/kernels/dataflow: reader_interleaved_qwen_chain.cpp") ;;
      *)
        echo "FAIL: the graft's sdpa directory differs from image ${image:7:12}'s by more than the chain reader:" >&2
        echo "$differences" >&2
        exit 1 ;;
    esac
    echo "ok    sdpa = image ${image:7:12}'s + reader_interleaved_qwen_chain.cpp"
  done
fi
if [ "$DRY" = 1 ]; then
  echo "### run: (cd $(printf '%q' "$G") && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > $(printf '%q' "$G/MANIFEST.sha256")"
else
  (cd "$G" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum) > "$G/MANIFEST.sha256"
fi
x rm -rf "$GRAFT"
x mv "$G" "$GRAFT"
if [ "$DRY" = 1 ]; then
  echo "### dry run: nothing built"
  exit 0
fi
so_sha=$(hsha "$GRAFT/_ttnncpp.so")
echo "### summary"
if [ "${K64J_KEEP_STAGED:-}" = "1" ]; then
  echo "ttbuild: KEPT STAGED - sources and build_Release carry the qwen factories and kernels"
else
  echo "ttbuild: sources restored (decode 3e0a69af, prefill fd8c0676, touched) and build_Release rebuilt from them"
fi
sha256sum "$GRAFT/_ttnncpp.so" "$GRAFT/_ttnn.so" "$BASE/_ttnncpp.so" "$BASE/_ttnn.so"
echo "graft files: $(wc -l < "$GRAFT/MANIFEST.sha256") (manifest $GRAFT/MANIFEST.sha256)"
if [ -n "${RESULTS:-}" ]; then
  {
    mkdir -p "$RESULTS"
    cp "$GRAFT/MANIFEST.sha256" "$RESULTS/k64j-MANIFEST.sha256"
    cp "$W"/*-qwen-strings.txt "$RESULTS/"
    printf 'graft=%s\nbase=%s\nK64J_TTNNCPP_SHA256=%s\nK64I_TTNNCPP_SHA256=%s\nfactory=%s\nwork=%s\n' \
      "$GRAFT" "$BASE" "$so_sha" "$base_so" "$FACTORY_K64J" "$W" > "$RESULTS/k64j-build-summary.txt"
  } || echo "WARN  could not copy the manifest and strings lists into $RESULTS" >&2
fi
echo "next: the card tests with CARD_B_ENV=WATCHER=1 KOPGRAFT64=$GRAFT EXPECT_TTNNCPP_SHA256=$so_sha"
echo "K64J_TTNNCPP_SHA256=$so_sha"
echo "### build_k64j done $(date -Is)"
