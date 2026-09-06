# Experimental GDN V-split prototype

## Status and ownership

All six generated kernel sources and the two-stage orchestration are implemented in
`gdn_vsplit.py`. This is an isolated, opt-in prototype, not an engine change or a
certified kernel. Only host tests and the read-only source audit have been run.
No device compilation, TTsim, hardware execution, CI dispatch, commit or push is
part of this work. The main agent owns simulator scheduling, the running hardware
recertification gate `34044312880`, and any later engine integration. Default
pathways and existing files remain unchanged.

The reported native B1 and T32 verifier timings are context, not measurements of
this prototype. No speedup or progress toward 200 committed tokens/s is claimed.

## Contracts

- Existing 1x2 mesh, at least 96 compute/storage cores per chip; intended for two
  P150A devices. Device identity and architecture must be checked by the central
  harness; this module checks mesh shape and core count, not board identity.
- One sequence, T in `1,2,4,8,16,32`; no independent batches, T>32, or arbitrary
  geometry. All six inputs are interleaved DRAM BF16 TILE tensors.
- QKV `[1,T,5120]`, beta/g `[1,T,24]`, immutable initial state
  `[1,24,128,128]`, z `[1,T,3072]`, norm weight `[1,1,128]`.
- Input/output buffers must not alias. The initial state is never written.
- Returns `(output, states, pre_norm)`: BF16 TILE `[1,T,3072]`, BF16 TILE
  `[T,24,128,128]`, and FP32 ROW_MAJOR `[T,1,96,32]`, all in DRAM.
  Shapes describe each chip's local TP2 tensors, as in the existing prototype.

### Stage 1: 96 recurrence workers

`Kt=4`, `Vt=1`, virtual H=96, RF=12. Worker `4*head+partition` owns one V32
partition of one native head. Q/K source head is `worker//12`; V tile column is
`64+worker`; beta and g select `worker//4` from the original 24-head scalar row.

Physical prefix-state tile address is
`token*384+(worker//4)*16+key_tile*4+worker%4`. The reader loads only token zero
from the initial state, gathering the four K tiles with stride four. The native
nonFNG CB30 feedback path rounds new state to BF16 between tokens. The writer
scatters four state tiles with the same stride and retains the native state layout.

The nonFNG compute source is exactly `gdn_multitoken.load_kernels(root, False)`'s
compute source. Its `qn @ new_h` result packs to FP32 CB19, not BF16, while state
CB18 and feedback CB30 remain BF16. The output accessor uses 128-byte row sticks;
state writes independently use `tb_state=get_tile_size(cb_sout)` (2048 bytes),
not `tb_io=get_tile_size(cb_out)` (4096 bytes). Scratch staging now explicitly
pushes/waits before popping its reserved page.

### Stage 2: 24 whole-head norm/gate workers

One worker per original head reads four consecutive full 128-byte FP32 bridge
pages in partition order. A dedicated FP32 CB5 staging tile receives each stick;
volatile word copies put its two 16-float halves into faces 0 and 1 of a zeroed
FP32 CB15 tile, row zero. Reader staging CB5 and writer assembly CB27 are disjoint.

Native weight loading supplies a full eight-tile BF16 CB30 block once. Native
compute converts the first four tiles into persistent FP32 CB31 and consumes the
entire initial block, retaining the native producer-handoff discipline. z uses
BF16 CB2 and the reader builds native FP32 ones in CB6.

The exact native FNG block after the output matmul is extracted verbatim, with
all its native helper functions. Stage 2 has Vt=4: `rowsum_k` reduces the original
four V tiles in the original order. No BF16 pre-norm conversion and no per-V32
normalization are introduced. The native BF16 round trips after RMS*w and SiLU(z)
remain in CB30, followed by FP32 multiplication and BF16 output packing.

The local native-derived writer zeroes four BF16 assembly tiles, places token
rows at the native face offsets, and writes four full tiles to pages
`head*4+tile`. There is no state writer or remote semaphore path in this stage.
For T<32, unused output rows remain zero.

## Circular-buffer budget

| Stage | Workers/chip | CB bytes/worker | Estimated static end with 111,488 reserved |
|---|---:|---:|---:|
| Recurrence | 96 | 282,624 | 394,112 |
| Norm/gate | 24 | 204,800 | 316,288 |

These are descriptor sums and arithmetic estimates using the user-confirmed
reserved amount, not compiler/linker or runtime L1 measurements. The existing
630,784-byte FNG plan is not reused. All recurrence Q/K input, normalized,
transpose, square and FP32 mirror capacities remain four. State capacities
shrink from 16 to four; V-only capacities shrink from four to one, with two-page
double rings where required. The second stage retains four-wide whole-head math
and eight-page native round-trip/output rings. `cb_plan` and the host audit
expose every CB's format and capacity.

At T32, prefix-state exports alone are 24 MiB per chip, and the FP32 bridge is
384 KiB per chip. Fourfold duplicated Q/K normalization, 96-worker placement,
DRAM traffic, a second program, and fences may outweigh the smaller per-worker
recurrence. Two-stage memory reuse and actual L1 placement need device validation.

## Pins and host-only checks

The generator pins the existing Python helper and uses its three native source
SHA256 checks plus unique/ordered section anchors. Missing, duplicate, reordered
or changed source fails closed. `execute` also freshly checks the helper's pinned
Blackhole packer and dataflow header hashes in the active `TT_METAL_HOME`
(default `/opt/tt-metal`), independently of the selected source root and without
the existing validator's cache. The evidence source tree lacks these runtime
headers: it is sufficient for host generation, not a runtime checkout. Execution
additionally requires a full matching active tt-metal installation. Drift must
be reviewed, not bypassed by disabling validation.

From the worktree root, these commands are host-only and write no generated files
or Python bytecode:

```powershell
py -3.11 -B -m unittest discover -s scripts/ci -p test_gdn_vsplit.py -v
py -3.11 -B scripts/ci/gdn_vsplit.py --root hardware-evidence.local/34009341359/qwen-hardware-inventory-34009341359/gdn-source
```

Set `GDN_VSPLIT_SOURCE_ROOT` to a matching source root if using another worktree.
Source-dependent tests explicitly skip if the pinned source is absent; mapping,
ABI and mocked orchestration tests still run. Kernel generation itself always
fails closed on missing or mismatched native source. Coverage
includes exhaustive state mapping, packed GQA/scalar mapping, FP32 bit-preserving
bridge layout, tiled output/padding, hash/anchor rejection, exact FNG extraction,
CB formats/capacities, compile/runtime ABI, and mocked two-chip orchestration.
Mocks do not establish C++ API compatibility, DMA behavior, numerical correctness
on the device, or deadlock freedom.

## Central validation handoff

1. Review the source audit and pins against the intended full tt-metal checkout.
   Confirm board identity, physical worker availability, 128-byte DRAM accessor
   pages and actual per-stage L1 placement. The current snapshot has not compiled
   the new C++ sources.
2. Main agent schedules the first simulator job. Import `gdn_vsplit` on the
   harness's module path and explicitly call
   `execute(mesh, qkv, beta, gate, initial, z=z, norm_w=norm_w, root=root, experimental=True)`.
   This API consumes existing device tensors; it never opens a mesh itself.
3. Start T1, then T2/4/8/16/32 on both chips. Compare every prefix state and gated
   row to serial exact native FNG, and inspect the FP32 bridge against the native
   pre-norm output before any BF16 packing. Cover nonzero initial states, distinct
   heads/partitions, both scalar columns 15/16, token rows 15/16, mixed signs,
   small norms, and all padding. Verify initial-state immutability and continuation
   from each exported prefix. Trace CB progress if any read/write hangs.
4. Specifically validate FP32 full-stick read/write behavior, BF16 state DMA
   strides, CB30 reader-to-compute ownership transition, BF16 round trips and
   full-head reduction order. Source equality is not numeric certification.
5. Only after central simulator approval and the separate hardware gate, schedule
   hardware correctness/continuation checks. Add a separately owned fenced/trace
   timing harness only after correctness. `execute` deliberately fences both
   programs for bridge visibility and lifetime safety; it is not a performance
   or trace-capture path. It performs no model integration or ownership changes.
