#!/usr/bin/env bash
# Compiles one of the P2P device kernels with the REAL RISC-V toolchain (the sfpi g++ of the tt-metal tree) and the flags tt-metal's JIT uses for a
# Blackhole BRISC kernel (-Wall -Werror, -ftt-no-dyninit, C++17 plus the tt extensions), behind the same include prelude as the firmware's brisck.cc.
# It is a front-end check, not a build: it catches the errors the JIT would raise at the first dispatch on a card (a wrong API call, a warning promoted
# to an error, a banned dynamic initialisation) without a device. It links nothing and runs nothing. Run it INSIDE the build container (build_kv_read.sh does):
#
#   bash rv_kernel_check.sh <directory holding the kernel> scatter|checksum "<compile-time args, comma separated>"
#
# The compile-time args are the kernel's own, accessor arguments included: each TensorAccessorArgs is (config, aligned page size) and the config of a DRAM
# interleaved buffer is 2 (IsDram), which selects the DRAM branch of the accessor, so pass 2 there. Exit status 0 = the kernel compiles.
set -eu
dir=$1
kernel=$2
args=$3
R=/opt/tt-metal
CC=$R/runtime/sfpi/compiler/bin/riscv-tt-elf-g++
test -x "$CC" || { echo "no RISC-V compiler at $CC" >&2; exit 2; }
test -s "$dir/qwen_p2p_$kernel.cpp" || { echo "no $dir/qwen_p2p_$kernel.cpp" >&2; exit 2; }
work=$(mktemp -d /tmp/rvcheck.XXXXXX)
trap 'rm -rf "$work"' EXIT
cat > "$work/wrap.cc" <<EOT
#include <unistd.h>
#include <cstdint>
#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#include "$dir/qwen_p2p_$kernel.cpp"
EOT
INC="-I. -I.. -I$R/ -I$R/ttnn -I$R/ttnn/cpp -I$R/tt_metal -I$R/tt_metal/hw/inc -I$R/tt_metal/tt-llk/common -I$R/tt_metal/hostdevcommon/api -I$R/tt_metal/api/"
INC="$INC -I$R/tt_metal/hw/ckernels/blackhole/metal/common -I$R/tt_metal/hw/ckernels/blackhole/metal/llk_io -I$R/tt_metal/hw/inc/internal/tt-1xx"
INC="$INC -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
INC="$INC -I$R/tt_metal/tt-llk/tt_llk_blackhole/common/inc -I$R/tt_metal/tt-llk/tt_llk_blackhole/llk_lib -I$R/runtime/sfpi/include"
FLAGS="-std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit -flto=auto -ffast-math -fno-finite-math-only -fsigned-zeros -fno-associative-math"
FLAGS="$FLAGS -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=multistatement-macros"
FLAGS="$FLAGS -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function -mcpu=tt-bh -fno-tree-loop-distribute-patterns -O2"
DEFS="-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DPROCESSOR_INDEX=0 -DCOMPILE_FOR_BRISC -DPROGRAMMABLE_CORE_TYPE=0 -DDISPATCH_MESSAGE_ADDR=0 -DKERNEL_BUILD"
DEFS="$DEFS -DARCH_BLACKHOLE -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DNOC_INDEX=0 -DNOC_MODE=0 -DNUM_DRAM_BANKS=8 -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3"
DEFS="$DEFS -DNUM_L1_BANKS=140 -DLOG_BASE_2_OF_NUM_L1_BANKS=7 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DKERNEL_COMPILE_TIME_ARGS=$args"
cd "$work"
# shellcheck disable=SC2086
nice -n 19 "$CC" $FLAGS $DEFS $INC -S wrap.cc -o wrap.s
echo "rv_kernel_check: $kernel compiles with the RISC-V toolchain"
# The same source once more WITHOUT link-time optimisation (an LTO assembly file holds no code), to read the code generated: the checksum's 32 x 32 -> 64 bit
# product must be mul + mulhu, never a library call (__muldi3 / __umulsidi3 would be an unresolved symbol at the JIT's link). -Warray-bounds is off for this
# pass only: a firmware header the wrapper includes trips it without LTO.
if [ "$kernel" = checksum ]; then
  # shellcheck disable=SC2086
  nice -n 19 "$CC" ${FLAGS/-flto=auto/} -Wno-array-bounds $DEFS $INC -S wrap.cc -o wrap_code.s
  grep -q '[[:space:]]mulhu[[:space:]]' wrap_code.s || { echo "rv_kernel_check: no mulhu in the checksum kernel" >&2; exit 1; }
  if grep -q '__muldi3\|__umulsidi3\|__mulsi3' wrap_code.s; then echo "rv_kernel_check: the checksum kernel calls a multiply library routine" >&2; exit 1; fi
  echo "rv_kernel_check: checksum multiplies with mul + mulhu"
fi
