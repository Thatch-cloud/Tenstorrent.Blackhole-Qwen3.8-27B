#!/usr/bin/env bash
# Apply the p2pmem patch to a checkout of tenstorrent/tt-kmd and build it against a kernel build tree, to prove that it applies and compiles.
# Nothing is installed or loaded: the checkout is exported to a scratch directory and built there.
#
#   build-check.sh <tt-kmd git checkout> <kernel build dir> [--no-p2pdma]
#
# <kernel build dir> is /lib/modules/$(uname -r)/build, or the unpacked linux-headers of the kernel the module is for.
# --no-p2pdma also builds the variant a kernel without CONFIG_PCI_P2PDMA gets (the module must still build, and publishes nothing).
set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
kmd=$(readlink -f "$1")
kdir=$(readlink -f "$2")
variant=${3:-}
here=$(cd "$(dirname "$0")" && pwd)
patch=$(ls "$here"/0001-*.patch)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git -C "$kmd" rev-parse --git-dir >/dev/null
git -C "$kmd" archive HEAD | tar -x -C "$work"
git -C "$work" init -q 2>/dev/null || true
(cd "$work" && git apply --check "$patch" && git apply "$patch")
echo "applied $(basename "$patch") on tt-kmd $(git -C "$kmd" describe --tags --always)"

if [ "$variant" = "--no-p2pdma" ]; then
  sed -i 's/#if IS_ENABLED(CONFIG_PCI_P2PDMA)/#if 0/' "$work/p2pmem.c"
fi

log="$work/build.log"
make -C "$work" -j"$(nproc)" KDIR="$kdir" W=1 >"$log" 2>&1 || { tail -40 "$log"; echo "BUILD FAILED"; exit 1; }
if grep -E 'p2pmem\.[ch]:.*(warning|error)' "$log"; then
  echo "p2pmem produced warnings"; exit 1
fi
ko="$work/tenstorrent.ko"
for param in p2pmem_devices p2pmem_offset_mb p2pmem_size_mb p2pmem_publish; do
  modinfo "$ko" | grep -q "^parm: *$param:" || { echo "missing module parameter $param"; exit 1; }
done
if [ "$variant" != "--no-p2pdma" ]; then
  nm "$ko" | grep -q 'U pci_p2pdma_add_resource' || { echo "the module does not call pci_p2pdma_add_resource"; exit 1; }
fi
echo "OK  $(modinfo -F vermagic "$ko")  sha256 $(sha256sum "$ko" | cut -c1-16)  (built in a scratch directory, not installed)"
