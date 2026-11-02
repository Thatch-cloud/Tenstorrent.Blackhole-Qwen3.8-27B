#!/usr/bin/env bash
# Build qwen_kv_read.so (version 4: the region reads of versions 1-3 and the P2P KV offload's device ops, see qwen_kv_read.cpp) for a serving image built on tt-metal
# 62443b62, in a THROWAWAY container of that image. The container has no device, no network, a CPU and a memory limit, and writes nothing on the machine that runs it:
# the sources go in on its stdin and the result comes back on its stdout. The module replaces no binary of the image (graft-so-drops-image-patches cannot apply); it links
# the image's own _ttnncpp.so / libtt_metal.so / libtt_stl.so / libtt-umd / libtracy and the nanobind static library, with the flags of the ttnn unity build in the image's
# build.ninja (minus link-time optimisation and -Werror, plus the tt-metal root on the include path for mesh_tensor_impl.hpp), compiled by the image's clang++-20.
#
#   KVR_IMAGE=<the serving image> bash optimisation/ttnn-op/kv_region_read/build_kv_read.sh [out dir]        (default ./opgraft-KVR4)
#
# The build is checked before it is kept (in the same container, no device):
#   1. ldd -r: no unresolved symbol that is not Python's;
#   2. the RISC-V front end (rv_kernel_check.sh, the image's sfpi, tt-metal's JIT flags) on all four device kernels;
#   3. python -c "import ttnn, qwen_kv_read": the version, every op and exception on ttnn, the kernels beside the module;
#   4. p2p_staging_mock.py on tt-metal's MOCK Blackhole cluster: the real JIT compiles the kernels, and the refusal matrix, the digest handles, warm and freeze run against
#      the real tt-metal (KVR_MOCK=0 skips it).
# It writes <out>: qwen_kv_read.so, qwen_kv_read.o, compile-flags.txt, the sources it was built from (qwen_kv_read.cpp, qwen_p2p_plan.hpp, the four kernels, rv_kernel_check.sh,
# p2p_staging_mock.py, this script), image-binaries.txt (the sha256 of the image libraries it linked), MANIFEST.sha256 and build.log. Built in <out>.partial and moved into
# place only after every check passed.
#
# Env:
#   KVR_IMAGE          REQUIRED. The serving image to build in and to check against. Nothing is replaced in it.
#   KVR_REMOTE         A command prefix that gives a shell on the machine that has docker, for example "ssh -o BatchMode=yes user@host". Unset: docker runs here.
#   KVR_CPUS           CPUs of the container (default 6)
#   KVR_MEMORY         memory limit of the container (default 8g)
#   KVR_MIN_AVAILABLE_GIB   the build refuses to start unless MemAvailable on the docker machine is at least this (default 20: a rig that also serves must not be starved)
#   KVR_TT_METAL_ROOT  the tt-metal tree in the image (default: the image's own TT_METAL_HOME)
#   KVR_SFPI_ROOT      the sfpi toolchain in the image (default /opt/tenstorrent/sfpi)
#   KVR_MOCK           1 (default) runs p2p_staging_mock.py; 0 skips it
#   KVR_DRY_RUN=1      print the docker command and run nothing
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=${1:-./opgraft-KVR4}
image=${KVR_IMAGE:?KVR_IMAGE must name the serving image to build in (a tt-metal 62443b62 image)}
remote=${KVR_REMOTE:-}
cpus=${KVR_CPUS:-6}
memory=${KVR_MEMORY:-8g}
min_gib=${KVR_MIN_AVAILABLE_GIB:-20}
metal=${KVR_TT_METAL_ROOT:-}
sfpi=${KVR_SFPI_ROOT:-/opt/tenstorrent/sfpi}
mock=${KVR_MOCK:-1}

# The source files of the build: the module, its pure-arithmetic header, the four device kernels it JIT-compiles at run time, and the checks.
KERNELS="qwen_p2p_scatter.cpp qwen_p2p_gather.cpp qwen_p2p_checksum.cpp qwen_p2p_digest_rows.cpp"
SOURCES="qwen_kv_read.cpp qwen_p2p_plan.hpp $KERNELS rv_kernel_check.sh p2p_staging_mock.py"
for f in $SOURCES; do test -s "$here/$f" || { echo "no $f beside this script" >&2; exit 2; }; done

# The script that runs INSIDE the container (it travels in the same tar as the sources). fd 3 is the real stdout (the result tar); stdout and stderr are the log.
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cat > "$stage/inner.sh" <<'INNER'
exec 3>&1 1>&2
set -eu
metal=${1:-${TT_METAL_HOME:-}}; sfpi=$2; mock=$3
test -n "$metal" || { echo "no tt-metal root: the image sets no TT_METAL_HOME and KVR_TT_METAL_ROOT is empty" >&2; exit 2; }
cd /tmp/kvr
test -d "$metal/build_Release" || { echo "no $metal/build_Release in this image" >&2; exit 2; }
test -x "$sfpi/compiler/bin/riscv-tt-elf-g++" || { echo "no RISC-V compiler in $sfpi" >&2; exit 2; }
B=$metal/build_Release
# the tree must be the one the module is written for (the 62443b62 host API: a program is created for a mesh, UMD headers live under tt-umd/)
grep -q "CreateProgram(distributed::MeshDevice& mesh_device)" "$metal/tt_metal/api/tt-metalium/host_api.hpp" || { echo "this tt-metal is not the 62443b62 generation (no CreateProgram(MeshDevice&))" >&2; exit 2; }
test -f "$metal/tt_metal/api/tt-metalium/mesh_event.hpp" || { echo "this tt-metal has no mesh_event.hpp" >&2; exit 2; }
echo "== libraries linked"
(cd "$B" && sha256sum ttnn/_ttnncpp.so lib/libtt_metal.so lib/libtt_stl.so lib/libtt-umd.so.0.* lib/libtracy.so.0.* ttnn/libnanobind-static-abi3.a "$metal/ttnn/ttnn/_ttnn.so") | tee /tmp/kvr/image-binaries.txt
cd "$B"
# 1. compile + link (the flags are ninja's own for the ttnn unity sources, minus LTO and -Werror)
FLAGS=$(ninja -t commands ttnn/_ttnn.so | grep -m1 "unity_0_cxx.cxx.o -c" | sed -e 's/^[^ ]* //' -e 's/ -MD .*//' -e 's/-flto=thin//' -e 's/ -Werror//' -e 's/-Dttnn_EXPORTS//')
test -n "$FLAGS" || { echo "no unity compile command in build.ninja" >&2; exit 2; }
echo "$FLAGS" > /tmp/kvr/compile-flags.txt
echo "== compile"
eval nice -n 10 /usr/bin/clang++-20 "$FLAGS" -fvisibility=hidden -I"$metal" -I/tmp/kvr -c /tmp/kvr/qwen_kv_read.cpp -o /tmp/kvr/qwen_kv_read.o
echo "== link"
umd=$(ls "$B"/lib/libtt-umd.so.0.* | head -1)
tracy=$(ls "$B"/lib/libtracy.so.0.* | head -1)
/usr/bin/clang++-20 -fPIC -O3 -shared -fuse-ld=lld -Wl,--gc-sections -Wl,-soname,qwen_kv_read.so -o /tmp/kvr/qwen_kv_read.so /tmp/kvr/qwen_kv_read.o \
  -Wl,-rpath,"$B/tt_metal:$B/ttnn:$B/lib:$B/tt_metal/third_party/umd/lib:$B/tt_stl:" \
  "$B/ttnn/_ttnncpp.so" "$B/ttnn/libnanobind-static-abi3.a" "$B/lib/libtt_metal.so" "$B/lib/libtt_stl.so" "$umd" "$tracy" -ldl
readelf -d /tmp/kvr/qwen_kv_read.so | grep -E "NEEDED|SONAME"
# 2. ldd -r: every symbol that is not Python's resolves against the image's own libraries
echo "== ldd -r"
unresolved=$(ldd -r /tmp/kvr/qwen_kv_read.so 2>&1 | grep "undefined symbol" | grep -v "Py\|_Py" || true)
test -z "$unresolved" || { echo "UNRESOLVED:"; echo "$unresolved"; exit 3; }
echo "unresolved non-Python symbols: 0"
# 3. the device kernels, with the real RISC-V toolchain and the JIT's flags (a front-end check, no device)
echo "== RISC-V kernel front-end checks"
export TT_METAL_ROOT=$metal SFPI_ROOT=$sfpi
bash /tmp/kvr/rv_kernel_check.sh /tmp/kvr scatter "1088,64,16,32,2,1088,2,65536"
bash /tmp/kvr/rv_kernel_check.sh /tmp/kvr gather "1088,64,16,32,2,1088,2,65536"
bash /tmp/kvr/rv_kernel_check.sh /tmp/kvr checksum "32768,64,2,65536,2,64"
bash /tmp/kvr/rv_kernel_check.sh /tmp/kvr digest_rows "1088,64,8,2,16,2,1088,2,64"
# 4. the module imports and carries the contract
echo "== import"
PYTHONPATH=/tmp/kvr:${PYTHONPATH:-} python3 -c '
import os, ttnn, qwen_kv_read
assert qwen_kv_read.QWEN_KV_READ_VERSION == "4", qwen_kv_read.QWEN_KV_READ_VERSION
ops = ("qwen_read_blocks", "qwen_block_bytes", "qwen_read_blocks_raw", "qwen_write_blocks_raw", "qwen_p2p_staging_alloc", "qwen_p2p_staging_info", "qwen_blocks_to_staging",
       "qwen_write_blocks_from_staging", "qwen_blocks_digest_async", "qwen_p2p_record_event", "qwen_p2p_warm", "qwen_p2p_freeze", "qwen_staging_checksum",
       "qwen_p2p_staging_write", "qwen_p2p_staging_read")
assert all(callable(getattr(ttnn, n)) for n in ops), "an op is missing on ttnn"
assert issubclass(ttnn.QwenP2PRefused, ValueError) and issubclass(ttnn.QwenP2PEnqueueError, RuntimeError)
d = os.path.dirname(qwen_kv_read.__file__)
assert all(os.path.getsize(os.path.join(d, k)) > 0 for k in ("qwen_p2p_scatter.cpp", "qwen_p2p_gather.cpp", "qwen_p2p_checksum.cpp", "qwen_p2p_digest_rows.cpp")), "a kernel is missing beside the module"
sig = lambda n: (getattr(ttnn, n).__doc__ or "").splitlines()[0]
assert "pages_per_block: int" in sig("qwen_blocks_to_staging") and "cq_id: int = 0" in sig("qwen_write_blocks_from_staging"), "the gather / scatter signature is not the frozen interface"
assert "tensors:" in sig("qwen_blocks_digest_async") and "DigestHandle" in sig("qwen_blocks_digest_async"), "the digest signature is not the frozen interface"
print("IMPORT-OK", qwen_kv_read.QWEN_KV_READ_VERSION)
'
strings /tmp/kvr/qwen_kv_read.so | grep -oE 'QWEN_[A-Z0-9_]+' | sort -u > /tmp/kvr/qwen-strings-module.txt
echo "QWEN_ strings in the module: $(tr '\n' ' ' < /tmp/kvr/qwen-strings-module.txt)"
# 5. the mock Blackhole cluster: the real JIT, the refusal matrix, the digest handles, warm / freeze (no device in this container)
if [ "$mock" = 1 ]; then
  echo "== mock device"
  cd /tmp/kvr
  desc=$metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_4xP150.yaml
  test -f "$desc" || { echo "no mock cluster descriptor $desc" >&2; exit 2; }
  PYTHONPATH=/tmp/kvr:${PYTHONPATH:-} timeout 900 python3 /tmp/kvr/p2p_staging_mock.py --cluster-desc "$desc" 2>&1 | grep -v '^20[0-9][0-9]-' | tee /tmp/kvr/mock.log
  grep -q "^MOCK-OK" /tmp/kvr/mock.log || { echo "the mock-device check failed" >&2; exit 4; }
fi
sha256sum /tmp/kvr/qwen_kv_read.so
cd /tmp/kvr && tar cf - qwen_kv_read.so qwen_kv_read.o compile-flags.txt image-binaries.txt qwen-strings-module.txt >&3
INNER

# The command that runs the container. Nothing is mounted: the sources and the script above arrive on stdin, in one tar.
unpack='mkdir -p /tmp/kvr && cd /tmp/kvr && tar xf - && exec bash /tmp/kvr/inner.sh "$@"'
docker_cmd="docker run -i --rm --network none --cpus $cpus --memory $memory --entrypoint bash $(printf '%q' "$image") -c $(printf '%q' "$unpack") bash $(printf '%q' "$metal") $(printf '%q' "$sfpi") $mock"
guard="a=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); [ \"\$a\" -ge $((min_gib * 1048576)) ] || { echo \"MemAvailable \$a kB is under $min_gib GiB: not starting the build\" >&2; exit 9; }"
run_command="$guard; exec $docker_cmd"

if [ "${KVR_DRY_RUN:-0}" = 1 ]; then
  echo "### the container command (run on ${remote:-this machine}):"
  echo "$run_command"
  echo "dry run: nothing built"
  exit 0
fi

work=$out.partial
rm -rf "$work"; mkdir -p "$work"
echo "build_kv_read: image $image, container limits $cpus CPUs / $memory, out $out" | tee "$work/build.log"
sources_tar=$stage/sources.tar
tar cf "$sources_tar" -C "$here" $SOURCES -C "$stage" inner.sh
if [ -n "$remote" ]; then
  # shellcheck disable=SC2086
  $remote "$run_command" < "$sources_tar" 2> >(tee -a "$work/build.log" >&2) | tar xf - -C "$work"
else
  bash -c "$run_command" < "$sources_tar" 2> >(tee -a "$work/build.log" >&2) | tar xf - -C "$work"
fi
for f in qwen_kv_read.so qwen_kv_read.o compile-flags.txt image-binaries.txt; do test -s "$work/$f" || { echo "build_kv_read: the container returned no $f: a check failed (see $work/build.log)" >&2; exit 1; }; done
for f in $SOURCES; do cp "$here/$f" "$work/"; done
cp "$here/$(basename "${BASH_SOURCE[0]}")" "$work/"
sha=$(sha256sum < "$work/qwen_kv_read.so" | cut -c1-64)
echo "KVR_SHA256=$sha" | tee -a "$work/build.log"
(cd "$work" && find . -type f ! -name MANIFEST.sha256 ! -name build.log | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
rm -rf "$out"; mv "$work" "$out"
echo "build_kv_read: OK $out qwen_kv_read.so $sha"
