"""Device-resident learned layer-zero MLP branch and independent stage checks."""

import math
import os

import draft_wide_tp
import tp_shapes
from draft_convolution import grouped_causal_convolution, convolution_reference
from draft_mlp import split_mlp_weights, swiglu_device, swiglu_reference
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance, rms_reference
from projection_rounding import grouped_projection_reference


DRAFT_BF8_FLAG = 'QWEN_FAST_DRAFT_BF8'
# QWEN_FAST_DRAFTER_BF16=1 keeps every drafter projection bfloat16 whatever QWEN_FAST_DRAFT_BF8 says (the serving image
# bakes DRAFT_BF8=1). Default off: the dtype is decided by DRAFT_BF8 exactly as before.
DRAFTER_BF16_FLAG = 'QWEN_FAST_DRAFTER_BF16'
# tp4/fx-wp6 (op-fusion programme, work package 6): three default-off drafter levers, strict 0 or 1 (draft_fusion_tp.py holds the full text). They are
# read here with a plain environment read and their modules are imported only when the flag is on, so a process with all three off runs exactly the
# ops below and does not even load the new files.
REDUCE_FLAG = 'QWEN_FAST_DRAFT_REDUCE'        # draft_reduce_tp: the gather-add chain's slices and adds as one launch
TAIL_FLAG = 'QWEN_FAST_DRAFT_TAIL'            # draft_tail_tp: SwiGLU and the residual tail as one launch each
GATEUP1_FLAG = 'QWEN_FAST_DRAFT_GATEUP1'      # draft_gateup_tp: gate and up as one matmul over a load-time concatenated weight
MM_GRID_FLAG = 'QWEN_FAST_DRAFT_MM_GRID'      # draft_mmgrid_tp: this branch's matmuls on program grids as wide as the device's, not the fixed 8-wide ones


def _lever(name):
    value = os.environ.get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def _levers_checked():
    """The four fx-wp6 flags, each strict, and no _AUDIT flag without its lever (an audited arm that audits nothing would pass): -> {flag: on}."""
    found = {name: _lever(name) for name in (REDUCE_FLAG, TAIL_FLAG, GATEUP1_FLAG, MM_GRID_FLAG)}
    for name, on in found.items():
        if _lever(name + '_AUDIT') and not on:
            raise ValueError('%s_AUDIT needs %s=1: the audit would compare nothing' % (name, name))
    return found


ENGAGED_MARKER = '[DRAFTER_BF16] engaged: draft projections upload as bfloat16 (QWEN_FAST_DRAFTER_BF16=1, overrides QWEN_FAST_DRAFT_BF8)'
_announced = []


def _announce_bf16():
    """The engaged marker, once per process (the dtype is asked for at each of the 36 projection uploads)."""
    if not _announced:
        _announced.append(True)
        print(ENGAGED_MARKER, flush=True)


def draft_projection_dtype(operations, environ=None):
    """The upload dtype of the draft's PROJECTION matrices - the five DFlash2 layers' q/k/v/o
    and gate/up/down, and the fc feature projection: bfloat8_b under QWEN_FAST_DRAFT_BF8=1,
    bfloat16 otherwise (the default, and every other draft tensor always: norms,
    convolution kernels and bases, the selector). Read at each upload, never cached."""
    environ = os.environ if environ is None else environ
    if environ.get(DRAFTER_BF16_FLAG) == '1':
        _announce_bf16()
        return operations.bfloat16
    return operations.bfloat8_b if environ.get(DRAFT_BF8_FLAG) == '1' else operations.bfloat16


def gate_up_columns_of(parameters):
    """Per-core output columns of the gate / up matmuls: ceil(shard tiles / 80 cores) from the prepared shards (4 at the
    pair, 2 at four cards); without shards (a fixture) the width's own value."""
    shards = parameters.get('shards')
    shape = getattr(shards[0][0], 'shape', None) if shards else None
    if shape:
        return math.ceil((shape[1] // 32) / 80)
    return 4 if tp_shapes.chip_count() == tp_shapes.PAIR else 2


def prepare_mlp_branch(operations, mesh, weights, convolution, retain):
    import torch

    shards = split_mlp_weights(*(weights[f'layers.0.mlp.{name}_proj.weight'] for name in ('gate', 'up', 'down')))
    norm_weight = convolution['layers.0.post_attention_layernorm.weight']
    conv_weight = convolution['layers.0.mlp_conv.kernel_projection.weight'].T.contiguous()
    base_weight = convolution['layers.0.mlp_conv.base_kernel']

    def upload(value, sharded=False, layout=None, dtype=None):
        return retain(operations.from_torch(value, device=mesh, dtype=operations.bfloat16 if dtype is None else dtype,
            layout=operations.TILE_LAYOUT if layout is None else layout, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=0) if sharded else operations.ReplicateTensorToMesh(mesh)))

    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    # QWEN_FAST_DRAFT_GATEUP1 (draft_gateup_tp.py): one (5120, 2 x 4352) gate|up weight per chip in place of the separate gate and up uploads, so the
    # branch runs one matmul for both. A shard the fused program cannot take keeps the separate uploads (a logged fall-back).
    fused_gate_up = False
    if _levers_checked()[GATEUP1_FLAG]:
        import draft_gateup_tp

        draft_gateup_tp.enabled()
        reason = draft_gateup_tp.shard_problem(shards)
        if reason is None:
            fused_gate_up = True
        else:
            draft_gateup_tp.fusion.fell_back(draft_gateup_tp.fusion.GATEUP1_FALLBACK, 'mlp', reason)

    def projections():
        # evaluated where the served upload sat (after the norm, the conv and the bases), so the flag-off allocation order is what it was
        if fused_gate_up:
            return [upload(draft_gateup_tp.fused_host_weight(shards), True, dtype=draft_projection_dtype(operations)),
                    upload(draft_gateup_tp.separate_host_weight(shards, 2), True, dtype=draft_projection_dtype(operations))]
        return [upload(torch.cat([rank[index] for rank in shards], dim=0), True,
                       dtype=draft_projection_dtype(operations)) for index in range(3)]

    return dict(operations=operations, mesh=mesh, source_weights=weights, source_convolution=convolution,
        shards=shards, norm_weight=norm_weight, conv_weight=conv_weight, base_weight=base_weight, kernel=kernel,
        device_norm=upload(norm_weight.reshape(1, 1, 160, 32), layout=operations.ROW_MAJOR_LAYOUT),
        device_conv=upload(conv_weight),
        bases=[upload(base_weight[phase, offset].reshape(1, 1, 1, 5120)) for phase in range(2) for offset in range(2)],
        device_projections=projections(), **(dict(gateup1=True) if fused_gate_up else {}))


def execute_mlp_branch(operations, mesh, collectives, hidden, weights, convolution, retain, *, parameters=None,
                       trace_safe=False, convolution_operation=None, boundaries=None, observe=None, quad=None):
    convolve = convolution_operation or grouped_causal_convolution
    # QWEN_FAST_PROPOSAL_AUDIT (dflash_device.ProposalAudit): `observe(name, tensor)` reads
    # a replicated intermediate back from both chips at the stage that made it. None,
    # the default and the only value with the audit off, adds nothing.
    def watch(name, value):
        if observe is not None:
            observe(name, value)
        return value

    seams = {} if boundaries is None else dict(boundaries=boundaries)
    # QWEN_FAST_QUAD_DRAFT (quad_draft.py): the 64-row block of four packed users, trace-owned only; None, the
    # default and the only value with the flag off, keeps every call below as it was.
    rows = 32
    if quad is not None:
        if not trace_safe or boundaries is None:
            raise ValueError('The quad draft pass is a trace-owned packed block')
        rows = quad.rows
    if tuple(hidden.shape) != (1, 1, rows, 5120) or hidden.dtype != operations.bfloat16:
        raise ValueError('A padded32-row BF16 hidden block is required')
    if parameters is None:
        parameters = prepare_mlp_branch(operations, mesh, weights, convolution, retain)
    if (parameters['operations'] is not operations or parameters['mesh'] is not mesh
            or parameters['source_weights'] is not weights or parameters['source_convolution'] is not convolution):
        raise ValueError('Prepared MLP parameters belong to a different mesh or learned layer')
    kernel = parameters['kernel']
    ownership = dict(retain_temporaries=retain) if trace_safe else {}

    levers = _levers_checked()
    mm_grid = levers[MM_GRID_FLAG]
    if mm_grid:
        import draft_mmgrid_tp

        draft_mmgrid_tp.enabled()

    def project(value, weight, grid, columns):
        if mm_grid:
            # QWEN_FAST_DRAFT_MM_GRID: the same program on a grid as wide as the device's (same cores, same per-core columns, same in0_block_w)
            grid = draft_mmgrid_tp.grid_for(operations, mesh, value, weight, grid, columns, rows, kernel)
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32, per_core_N=columns,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        return retain(operations.matmul(value, weight, dtype=operations.float32, program_config=program,
            compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))

    normalized = watch('normalized', retain(draft_wide_tp.rms_norm(operations, hidden, epsilon=1e-6, weight=parameters['device_norm'],
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG, site='mlp')))
    projected = project(normalized, parameters['device_conv'], (8, 5), 1)
    rounded = watch('conv-kernels', retain(operations.typecast(projected, operations.bfloat16)))
    dynamic = [retain(operations.slice(rounded, (0, 0, 0, offset * 320), (1, 1, rows, (offset + 1) * 320)))
        for offset in range(4)]
    bases = parameters['bases']
    prepared = watch('conv-in', retain(convolve(operations, mesh, normalized, dynamic[:2], bases[:2], fp32_intermediates=True, **ownership, **seams)))
    # per-core output columns of the gate and up matmuls on the 8x10 grid: ceil(tiles / 80) - 4 at the pair (272 tiles),
    # 2 at four cards (136); only which core owns each output tile moves, never the K reduction
    gate_up_columns = gate_up_columns_of(parameters)
    tail = levers[TAIL_FLAG]
    if tail:
        import draft_tail_tp

        draft_tail_tp.enabled()
    if parameters.get('gateup1'):
        # QWEN_FAST_DRAFT_GATEUP1: ONE matmul for gate and up (per-core columns twice the served ones), output columns [gate | up]
        import draft_gateup_tp

        fused = draft_gateup_tp.project_fused(operations, mesh, parameters, prepared, project, rows)
        projections = [fused]
        if tail:
            activation = draft_tail_tp.swiglu_fused(operations, mesh, fused, retain, served=swiglu_device)
        else:
            width = tuple(fused.shape)[3] // 2
            halves = [retain(operations.slice(fused, (0, 0, 0, offset), (1, 1, rows, offset + width))) for offset in (0, width)]
            activation = swiglu_device(operations, *halves, retain)
    else:
        projections = [project(prepared, parameters['device_projections'][index], (8, 10), gate_up_columns)
            for index in range(2)]
        # QWEN_FAST_DRAFT_TAIL (draft_tail_tp.py): the nine SwiGLU launches as one
        activation = (draft_tail_tp.swiglu(operations, mesh, *projections, retain, served=swiglu_device) if tail
                      else swiglu_device(operations, *projections, retain))
    partial = project(activation, parameters['device_projections'][-1], (8, 10), 2)
    gather = gather_add_projection if quad is None else quad.gather_add_projection
    if levers[REDUCE_FLAG]:
        # QWEN_FAST_DRAFT_REDUCE (draft_reduce_tp.py): the four slices and three adds after the gather as one launch
        import draft_reduce_tp

        gather = draft_reduce_tp.choose(gather, quad, 'mlp')
    reduced = watch('reduced', retain(gather(operations, mesh, collectives, partial, **ownership,
        **(dict(observe=observe) if observe is not None else {}))))
    rounded_output = retain(operations.typecast(reduced, operations.bfloat16))
    finished = watch('conv-out', retain(convolve(operations, mesh, rounded_output, dynamic[2:], bases[2:], fp32_intermediates=True, **ownership, **seams)))
    if tail:
        # the residual tail (two typecasts, an fp32 add, a typecast) as one launch
        output = watch('output', draft_tail_tp.residual(operations, mesh, finished, hidden, retain))
    else:
        wide_finished = retain(operations.typecast(finished, operations.float32))
        wide_hidden = retain(operations.typecast(hidden, operations.float32))
        summed = retain(operations.add(wide_finished, wide_hidden, dtype=operations.float32))
        output = watch('output', retain(operations.typecast(summed, operations.bfloat16)))
    return dict(hidden=hidden, normalized=normalized, conv_projection=projected, dynamic=dynamic, prepared=prepared,
        projections=projections, activation=activation, partial=partial, reduced=reduced, finished=finished,
        output=output, shards=parameters['shards'], norm_weight=parameters['norm_weight'],
        conv_weight=parameters['conv_weight'], base_weight=parameters['base_weight'],
        **(dict(gateup1=True) if parameters.get('gateup1') else {}))


def validate_projection(actual, inputs, weight, *, stage, failure_capture=None, checkpoint=None, layer=0, chip=0):
    import torch

    expected = grouped_projection_reference(inputs, weight, destination_rounding=True, fidelity_span=32)
    close = torch.isclose(actual.double(), expected, rtol=1e-4, atol=1e-4)
    if failure_capture is not None and not close.all():
        torch.save(dict(activation=inputs.contiguous().clone(), weight=weight.contiguous().clone(),
            actual=actual.double().clone(), reference=expected.clone(),
            ordinary_reference=inputs.double() @ weight.double(),
            mismatches=(~close).nonzero(), stage=stage,
            chip=chip, checkpoint=checkpoint, layer=layer, fidelity_span=32), failure_capture)
        print(f'Connected MLP {stage} failure capture: {failure_capture}', flush=True)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-4,
        msg=lambda message: f'Layer {layer} chip {chip} {stage}: {message}')
    return float((actual.double() - expected).abs().max())


def validate_mlp_branch(state, host, chip, *, failure_capture=None, checkpoint=None, layer=0):
    import torch

    def exact(actual, expected, stage):
        if not torch.equal(actual, expected):
            raise AssertionError(f'Connected MLP {stage} must be exact')

    def projection(actual, inputs, weight, stage):
        return validate_projection(actual, inputs, weight, stage=stage, failure_capture=failure_capture,
            checkpoint=checkpoint, layer=layer, chip=chip)

    hidden = host(state['hidden'], chip)
    normalized = host(state['normalized'], chip)
    norm_ulps = int(bf16_ulp_distance(normalized, rms_reference(hidden, state['norm_weight'])).max())
    if norm_ulps > 2:
        raise AssertionError('Connected MLP normalization exceeds two BF16 ULPs')
    conv_projection = host(state['conv_projection'], chip)
    conv_error = projection(conv_projection, normalized, state['conv_weight'], 'convolution')
    dynamic = conv_projection.bfloat16().split(320, dim=-1)
    for actual, expected in zip(state['dynamic'], dynamic, strict=True):
        exact(host(actual, chip), expected, 'dynamic kernel')
    bases = state['base_weight']
    prepared = host(state['prepared'], chip)
    exact(prepared, convolution_reference(normalized, dynamic[:2],
        [bases[0, offset].reshape(1, 1, 1, 5120) for offset in range(2)]), 'prepare convolution')
    projections = [host(value, chip) for value in state['projections']]
    if state.get('gateup1'):
        # QWEN_FAST_DRAFT_GATEUP1: one fused (rows, gate | up) projection - read its two column halves as the gate and the up projection
        half = projections[0].shape[-1] // 2
        projections = [projections[0][..., :half], projections[0][..., half:]]
    projection_errors = [projection(value[..., :8, :], prepared[..., :8, :], state['shards'][chip][index], ('gate', 'up')[index])
        for index, value in enumerate(projections)]
    activation = host(state['activation'], chip)
    activation_ulps = int(bf16_ulp_distance(activation, swiglu_reference(*projections)).max())
    if activation_ulps > 2:
        raise AssertionError('Connected MLP activation exceeds two BF16 ULPs')
    down_error = projection(host(state['partial'], chip)[..., :8, :], activation[..., :8, :], state['shards'][chip][2], 'down')
    reduced = host(state['reduced'], chip)
    exact(reduced, host(state['partial'], 0) + host(state['partial'], 1), 'fabric sum')
    finished = host(state['finished'], chip)
    exact(finished, convolution_reference(reduced.bfloat16(), dynamic[2:],
        [bases[1, offset].reshape(1, 1, 1, 5120) for offset in range(2)]), 'finish convolution')
    exact(host(state['output'], chip), (finished.float() + hidden.float()).bfloat16(), 'residual')
    return dict(chip=chip, stage='connected_mlp', norm_ulps=norm_ulps, activation_ulps=activation_ulps,
        conv_projection_error=conv_error, gate_up_errors=projection_errors, down_error=down_error,
        prepare_exact=True, finish_exact=True, residual_exact=True, sum_exact=True)
