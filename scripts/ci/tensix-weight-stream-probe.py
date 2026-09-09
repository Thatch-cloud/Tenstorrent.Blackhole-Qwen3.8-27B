"""Simulator-only compressed-weight transport; no matmul, bandwidth or TG claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tensix_weight_stream import execute_stream, prepare_stream, stream_geometry
from tensix_weight_packet import packet_runtime


def raw_copy_program(operations, source, destination, geometry):
    workers = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 8), operations.CoreCoord(7, 8))])
    scratch = operations.CBDescriptor(total_size=geometry['tile_bytes'], core_ranges=workers,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=0,
            data_format=getattr(operations, geometry['dtype']), page_size=geometry['tile_bytes'])])
    program = operations.MeshProgramDescriptor()
    for chip, (source_shard, destination_shard) in enumerate(zip(operations.get_device_tensors(source),
            operations.get_device_tensors(destination), strict=True)):
        arguments = operations.RuntimeArgs()
        for worker in range(8):
            arguments[worker][8] = [source_shard.buffer_address(), destination_shard.buffer_address(),
                geometry['rows'] // 32 * (geometry['width'] // 32), worker]
        compile_args = [geometry['tile_bytes']]
        for value in (source_shard, destination_shard):
            compile_args.extend(operations.TensorAccessorArgs(value).get_compile_time_args())
        kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('tensix_weight_stream_raw.cpp')),
            core_ranges=workers, compile_time_args=compile_args, runtime_args=arguments,
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                noc=operations.NOC.RISCV_0_default))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=[kernel], cbs=[scratch])
    return program


def hashes(root, paths):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection', choices=('gate', 'down'), required=True)
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--producers', type=int, choices=(8, 16), default=8)
    parser.add_argument('--single-packet', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    geometry = stream_geometry(options.projection, options.blocks, options.producers)
    import torch
    import ttnn

    native = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__, geometry=geometry,
        operand_scope='Two synthetic compressed weight patterns, independent data on each chip',
        sources=hashes(Path(__file__).parent, ('tensix-weight-stream-probe.py', 'tensix_weight_stream.py',
            'tensix_weight_packet.py', 'tensix_weight_packet_reader.cpp',
            'tensix_weight_stream_reader.cpp', 'tensix_weight_stream_writer.cpp', 'tensix_weight_stream_sink.cpp',
            'tensix_weight_stream_raw.cpp', 'tiny_tile_matmul.py', 'attention_batch.py', 'feature_projection.py',
            'gdn_multitoken_conv.py')),
        native_sources=hashes(native, ('tt_metal/hw/inc/api/remote_circular_buffer.h',
            'tt_metal/hw/inc/api/dataflow/dataflow_api.h', 'tt_metal/hw/inc/api/tensor/tensor_accessor.h',
            'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h',
            'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so',
            'tt_metal/impl/buffers/global_circular_buffer.cpp', 'ttnn/core/global_circular_buffer.cpp',
            'ttnn/cpp/ttnn-nanobind/program_descriptors.cpp')),
        arm_policies=['generic', 'single-packet' if options.single_packet else 'generic'],
        target_integrated=False, eligible_for_hardware=False, reader_engagements=[],
        eager_checks=[], replay_checks=[], negative_controls=[])
    mesh = None
    owned, pipelines, traces = [], [], []
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage, projection=options.projection, blocks=options.blocks)), flush=True)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3)
        dtype = getattr(ttnn, geometry['dtype'])
        patterns = [torch.randn((1, 1, geometry['rows'], 2 * geometry['width']),
            generator=torch.Generator().manual_seed(seed)).bfloat16() for seed in (3827, 3828)]
        payloads = [ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in patterns]
        source = ttnn.from_torch(patterns[0], device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        owned.append(source)
        output_shape = (1, 1, geometry['rows'] // 32 * (geometry['width'] // 32), geometry['tile_bytes'] // 4)
        for unused in range(3):
            owned.append(ttnn.from_torch(torch.zeros(output_shape, dtype=torch.int32), device=mesh, dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        reference = owned[1]
        for arm, output in enumerate(owned[2:]):
            operations = packet_runtime(ttnn, report['reader_engagements']) if options.single_packet and arm == 1 else ttnn
            pipelines.append(prepare_stream(operations, mesh, source, output, geometry))
        if len(report['reader_engagements']) != (2 if options.single_packet else 0):
            raise AssertionError('Exactly one candidate reader per chip must be engaged')
        direct_program = raw_copy_program(ttnn, source, reference, geometry)
        bindings = [addresses(ttnn, value) for value in owned]
        if any(len({binding[chip] for binding in bindings}) != len(owned) for chip in range(2)):
            raise AssertionError('Borrowed input and all three outputs must be disjoint')
        def upload(pattern):
            ttnn.copy_host_to_device_tensor(payloads[pattern], source)
            ttnn.synchronize_device(mesh)
        def host(value, chip):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()
        def direct():
            ttnn.generic_op([source, reference], direct_program)
            ttnn.synchronize_device(mesh)
            return [host(reference, chip) for chip in range(2)]
        references = []
        progress('allocated')
        for pattern in range(2):
            upload(pattern)
            expected = direct()
            if torch.equal(expected[0], expected[1]):
                raise AssertionError('Independent chip inputs unexpectedly match')
            references.append(expected)
            for arm, prepared in enumerate(pipelines):
                execute_stream(ttnn, prepared)
                ttnn.synchronize_device(mesh)
                unchanged = direct()
                for chip in range(2):
                    if not torch.equal(host(prepared.output, chip), expected[chip]):
                        raise AssertionError('Tensix transport changed compressed weight words')
                    if not torch.equal(unchanged[chip], expected[chip]):
                        raise AssertionError('Tensix transport mutated borrowed compressed weights')
                    report['eager_checks'].append(dict(pattern=pattern, arm=arm, chip=chip,
                        exact_packed_words=True, inputs_unchanged=True))
            progress(f'eager_{pattern}_complete')
        if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
            raise AssertionError('Changed-input patterns must differ on both chips')
        upload(0)
        for prepared in pipelines:
            trace, unused_output = capture_operation(ttnn, mesh, lambda: execute_stream(ttnn, prepared))
            traces.append(trace)
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            for arm, (trace, prepared) in enumerate(zip(traces, pipelines, strict=True)):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                unchanged = direct()
                if [addresses(ttnn, value) for value in owned] != bindings:
                    raise AssertionError('Captured source or sink bindings changed')
                for chip in range(2):
                    if not torch.equal(host(prepared.output, chip), references[pattern][chip]):
                        raise AssertionError('Changed-input trace replay changed compressed words')
                    if not torch.equal(unchanged[chip], references[pattern][chip]):
                        raise AssertionError('Trace replay mutated borrowed compressed weights')
                    report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, arm=arm, chip=chip,
                        exact_packed_words=True, inputs_unchanged=True, bindings_stable=True))
                if repetition == 0:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for chip in range(2):
                        actual = host(prepared.output, chip)
                        if (not torch.equal(actual, references[0][chip])
                                or torch.equal(actual, references[1][chip])):
                            raise AssertionError('Stale-input negative control failed')
                        report['negative_controls'].append(dict(arm=arm, chip=chip, stale_detected=True))
            progress(f'replay_{repetition}_complete')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in reversed(traces):
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
                pipelines.clear()
                prepared = None
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
            report['sources_after'] = hashes(Path(__file__).parent, report['sources'])
            report['native_sources_after'] = hashes(native, report['native_sources'])
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                report['passed'] = False
                raise ValueError('Stream probe or native sources changed during execution')
            progress('complete' if report['passed'] else 'failed')
        finally:
            if not report['closed_cleanly']:
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
