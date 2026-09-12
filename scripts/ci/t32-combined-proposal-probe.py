"""Complete learned T32 proposal replay with synthetic cached K/V; no target-prefill or TG qualification."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from dspark_backbone_mesh import PARAMETERS, pack_parameter
from dspark_hardware_gate import digest
from dspark_intake import FILES
from dspark_layer import SPECIFICATIONS
from dspark_rope_tables import DSparkRotary
from dspark_t32_prepared import PreparedDSparkProposal
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from native_draft_sdpa import run_precise_probe
from sim_memory_budget import require_clean
from t32_attention_admission import require_active
from t32_ci_runtime import snapshot
from t32_sim_target import load as load_target


def sources():
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.iterdir())
        if path.suffix in ('.py', '.cpp', '.sh')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'target', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--publication-only', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    run_precise_probe(__file__)
    admission = require_active()
    if options.output.exists() or digest(options.config) != FILES['config.json'][1]:
        raise ValueError('Fresh report and pinned DSpark rotary configuration required')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    report = dict(passed=False, closed_cleanly=False, scope=__doc__, context=4096, capacity=4384,
        proposals=31, learned_layers=5, full_request_qualified=False, numerical_oracle_qualified=False,
        attention=admission, sources=sources(), resources_before=snapshot(), checks=[])
    if options.publication_only:
        report.update(scope='Learned captured history projection with synthetic feature taps; no target request or TG',
            proposals=0, publication_only=True)
    owned, mesh, prepared = [], None, None

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        progress('open_mesh')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)

        def upload(value, sharded=False, row_major=False):
            tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
                if sharded else ttnn.ReplicateTensorToMesh(mesh))
            owned.append(tensor)
            return tensor

        if not options.publication_only:
            progress('load_target_embedding_head')
            target, report['target_weights'] = load_target(ttnn, mesh, options.target, owned)
        parameters = {}
        with VerifiedWeights(options.checkpoint) as reader:
            report['draft_weight_hashes'] = reader.fingerprints()
            for name in PARAMETERS:
                progress('load_' + name)
                value, sharded = pack_parameter(name, reader.tensor(name))
                parameters[name] = upload(value, sharded)
                del value
            if not options.publication_only:
                predecessor = upload(reader.tensor('markov_head.markov_w1.weight').reshape(1, 1, 248320, 256), row_major=True)
                successor = upload(reader.tensor('markov_head.markov_w2.weight').T.contiguous().reshape(1, 1, 256, 248320))
        rotary = DSparkRotary(json.loads(options.config.read_text()))
        generator = torch.Generator().manual_seed(383932)
        if options.publication_only:
            from dspark_publication_trace import PreparedHistoryProjection

            layers = tuple({name: parameters[f'layers.{layer}.{name}'] for name in SPECIFICATIONS}
                for layer in range(5))
            previous = None
            for ordinal, position in enumerate((4096, 4111, 4128)):
                progress('publication_projection_' + str(position))
                features = tuple(upload((.1 * torch.randn(2, 1, 32, 2560,
                    generator=generator)).bfloat16(), True) for tap in range(5))
                tables = tuple(upload(value) for value in rotary.tables(position, 32))
                if prepared is None:
                    prepared = PreparedHistoryProjection(ttnn, mesh, collectives, parameters, layers,
                        features, tables, audit=True)
                else:
                    prepared.project(features, tables)
                actual = prepared.snapshot(prepared.outputs)
                if previous is not None and all(torch.equal(left, right)
                        for left, right in zip(actual, previous, strict=True)):
                    raise AssertionError('Changed publication inputs produced entirely stale outputs')
                previous = actual
                report['checks'].append(dict(position=position, tensors=len(actual), exact=True))
            report['replay_checks'] = prepared.checks
            report['passed'] = True
            return
        history = []
        for layer in range(5):
            pair = []
            for operand in range(2):
                value = (.1 * torch.randn(2, 4, 4384, 128, generator=generator)).bfloat16()
                value[:, :, 4096:] = 0
                pair.append(upload(value, True))
            history.append(tuple(pair))
        device = SimpleNamespace(operations=ttnn, mesh=mesh, collectives=collectives, target=target,
            closed=False, max_drafts=31, position=4096, rotary=rotary,
            parameters=parameters, predecessor=predecessor, successor=successor,
            layer_weights=tuple({name: parameters[f'layers.{layer}.{name}'] for name in SPECIFICATIONS}
                for layer in range(5)),
            history=SimpleNamespace(capacity=4384, layers=tuple(history), spare_layers=(), pending=None))
        progress('complete_proposal_eager_warmup')
        prepared = PreparedDSparkProposal(device, 10, audit=True, defer_capture=True)
        progress('complete_proposal_capture')
        prepared.capture()
        for anchor in (20, 10):
            progress('complete_proposal_changed_anchor_' + str(anchor))
            tokens = prepared.propose(anchor, 31)
            report['checks'].append(dict(anchor=anchor, tokens=list(tokens), exact=True))
        report['replay_checks'] = prepared.checks
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if prepared is not None:
                prepared.close()
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['sources_after'] = sources()
            if report['sources_after'] != report['sources']:
                raise ValueError('Proposal source changed during simulation')
            report['attention_after'] = require_active()
            if report['attention_after'] != report['attention']:
                raise ValueError('Proposal attention runtime changed during simulation')
            report['resources_after'] = snapshot()
            require_clean(report['resources_before'], report['resources_after'])
            report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = str(error)
            raise
        finally:
            progress('complete' if report['passed'] and report['closed_cleanly'] else 'failed')


if __name__ == '__main__':
    main()
