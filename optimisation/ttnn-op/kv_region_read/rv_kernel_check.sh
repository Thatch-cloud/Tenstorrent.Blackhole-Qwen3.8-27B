#!/usr/bin/env bash
# Compiles one of the P2P device kernels with the REAL RISC-V toolchain (the sfpi g++) and the flags tt-metal's JIT uses for a Blackhole BRISC kernel (-Wall -Werror,
# -ftt-no-dyninit, C++17 plus the tt extensions, -Os, link-time optimisation), behind the same include prelude as the firmware's brisck.cc.
# It is a front-end check, not a build: it catches the errors the JIT would raise at the first dispatch on a card (a wrong API call, a warning promoted
# to an error, a banned dynamic initialisation) without a device. It links nothing and runs nothing. Run it INSIDE the build container (build_kv_read.sh does):
#
#   bash rv_kernel_check.sh <directory holding the kernel> scatter|gather|checksum|digest_rows "<compile-time args, comma separated>"
#
# The compile-time args are the kernel's own, accessor arguments included: each TensorAccessorArgs is (config, aligned page size) and the config of a DRAM
# interleaved buffer is 2 (IsDram), which selects the DRAM branch of the accessor, so pass 2 there. Exit status 0 = the kernel compiles.
#
# Env: TT_METAL_ROOT  the tt-metal tree the image was built from (default /opt/tt-metal; the Gemma serving image keeps it under its own user's home)
#      SFPI_ROOT      the sfpi toolchain (default $TT_METAL_ROOT/runtime/sfpi; the Gemma image installs it as /opt/tenstorrent/sfpi)
set -eu
dir=$1
kernel=$2
args=$3
R=${TT_METAL_ROOT:-/opt/tt-metal}
SFPI=${SFPI_ROOT:-$R/runtime/sfpi}
CC=$SFPI/compiler/bin/riscv-tt-elf-g++
test -x "$CC" || { echo "no RISC-V compiler at $CC" >&2; exit 2; }
test -s "$dir/qwen_p2p_$kernel.cpp" || { echo "no $dir/qwen_p2p_$kernel.cpp" >&2; exit 2; }
work=$(mktemp -d /tmp/rvcheck.XXXXXX)
trap 'rm -rf "$work"' EXIT
# The prelude of tt_metal/hw/firmware/src/tt-1xx/brisck.cc. The headers that exist only in newer trees are included only where the tree has them.
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
#if __has_include("api/debug/kernel_profiler.hpp")
#include "api/debug/kernel_profiler.hpp"
#endif
#if __has_include("internal/profiler/noc_debugging_profiler.hpp")
#include "internal/profiler/noc_debugging_profiler.hpp"
#endif
#if __has_include("internal/debug/stack_usage.h")
#include "internal/debug/stack_usage.h"
#endif
#include "$dir/qwen_p2p_$kernel.cpp"
EOT
# The include directories of the JIT (build.cpp's list, then the Blackhole HAL's): a directory a tree does not have is ignored by the compiler.
INC="-I. -I.. -I$R/ -I$R/ttnn -I$R/ttnn/cpp -I$R/tt_metal -I$R/tt_metal/hw/inc -I$R/tt_metal/tt-llk/common -I$R/tt_metal/tt-llk/tools/include -I$R/tt_metal/hostdevcommon/api -I$R/tt_metal/api/"
INC="$INC -I$R/tt_metal/hw/ckernels/blackhole/metal/common -I$R/tt_metal/hw/ckernels/blackhole/metal/llk_io -I$R/tt_metal/hw/inc/internal/tt-1xx"
INC="$INC -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc"
INC="$INC -I$R/tt_metal/tt-llk/tt_llk_blackhole/common/inc -I$R/tt_metal/tt-llk/tt_llk_blackhole/llk_lib -I$R/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/hal"
INC="$INC -I$R/tt_metal/hw/firmware/src/tt-1xx -I$SFPI/include"
# JIT flags: common_flags + cflags of build.cpp (JitBuildEnv), Blackhole's -mcpu / -fno-tree-loop-distribute-patterns, and the default optimisation level of a data movement kernel (Os).
FLAGS="-std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit -flto=1 -flto-partition=one -ffast-math -fno-finite-math-only -fsigned-zeros -fno-associative-math"
FLAGS="$FLAGS -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=multistatement-macros"
FLAGS="$FLAGS -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function --param=min-pagesize=0"
FLAGS="$FLAGS -mcpu=tt-bh -fno-tree-loop-distribute-patterns -Os"
DEFS="-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DPROCESSOR_INDEX=0 -DCOMPILE_FOR_BRISC -DPROGRAMMABLE_CORE_TYPE=0 -DDISPATCH_MESSAGE_ADDR=0 -DKERNEL_BUILD"
DEFS="$DEFS -DARCH_BLACKHOLE -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DNOC_INDEX=0 -DNOC_MODE=0 -DNUM_DRAM_BANKS=8 -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3"
DEFS="$DEFS -DNUM_L1_BANKS=140 -DLOG_BASE_2_OF_NUM_L1_BANKS=7 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DKERNEL_COMPILE_TIME_ARGS=$args"
cd "$work"
# shellcheck disable=SC2086
nice -n 19 "$CC" $FLAGS $DEFS $INC -S wrap.cc -o wrap.s
echo "rv_kernel_check: $kernel compiles with the RISC-V toolchain"
# The same source once more WITHOUT link-time optimisation (an LTO assembly file holds no code), to read the code generated: the checksum's 32 x 32 -> 64 bit
# product must be mul + mulhu, never a library call (__muldi3 / __umulsidi3 would be an unresolved symbol at the JIT's link), and no kernel may divide or take a remainder by
# calling a library routine (__udivsi3 / __umodsi3 and friends: the core has the M extension, so a division is one instruction, and a call means the code was not compiled for it).
# -Warray-bounds is off for this pass only: a firmware header the wrapper includes trips it without LTO.
# shellcheck disable=SC2086
nice -n 19 "$CC" ${FLAGS/-flto=1 -flto-partition=one/} -Wno-array-bounds $DEFS $INC -S wrap.cc -o wrap_code.s
if grep -q '__muldi3\|__umulsidi3\|__mulsi3\|__udivsi3\|__umodsi3\|__divsi3\|__modsi3\|__udivdi3\|__umoddi3\|__divdi3\|__moddi3' wrap_code.s; then
  echo "rv_kernel_check: the $kernel kernel calls a multiply or divide library routine" >&2
  exit 1
fi
echo "rv_kernel_check: $kernel calls no multiply or divide library routine"
if [ "$kernel" = checksum ] || [ "$kernel" = digest_rows ]; then
  grep -q '[[:space:]]mulhu[[:space:]]' wrap_code.s || { echo "rv_kernel_check: no mulhu in the $kernel kernel" >&2; exit 1; }
  echo "rv_kernel_check: $kernel multiplies with mul + mulhu"
fi
