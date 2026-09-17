"""Complete loaded-model T32 hardware correctness audit; no throughput qualification."""

import argparse
import json
import os
import time
from pathlib import Path

from dspark_hardware_gate import digest
from dspark_intake import FILES
from dspark_pipeline import PARAMETERS, pack_parameter
from dspark_layer import SPECIFICATIONS
from dspark_rope_tables import DSparkRotary
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from request_host_health import snapshot
from t32_score_composition import audit as composition_audit
from target_t32_attention_gate import qualify_request
from t32_score_hardware import environment


def sources():
    roots = (Path(__file__).parent, Path('/speculative-decoding/harness'))
    return {str(path): digest(path) for root in roots for path in sorted(root.rglob('*'))
        if path.suffix in ('.py', '.cpp', '.sh')}


def load_rotary(path):
    return DSparkRotary(json.loads(path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'target', 'output', 'proposal-evidence', 'score-evidence', 'attention-evidence', 'commit-evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    options = parser.parse_args()
    environment()
    require_projection_environment(os.environ, True)
    if options.output.exists() or digest(options.config) != FILES['config.json'][1]:
        raise ValueError('Fresh output and pinned draft configuration required')
    directory = Path(__file__).parent
    admission = dict(proposal=composition_audit(directory, options.proposal_evidence, options.score_evidence),
        target_attention=qualify_request(options.attention_evidence, position=4096, remaining=64,
            hardware_mask_compatibility=True))
    from fused_t16_admission import qualify_simulator
    admission['target_mlp'] = qualify_simulator()
    from t32_commit_gate import qualify as qualify_commit
    admission['target_commit'] = qualify_commit(options.commit_evidence)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('target_loader_metadata', directory / 'dspark-target-hardware.py')
    metadata = module_from_spec(spec)
    spec.loader.exec_module(metadata)
    root = Path(os.environ['TT_METAL_HOME'])
    actual = {name: digest(root / name) for name in metadata.TARGET_SOURCES}
    if actual != metadata.TARGET_SOURCES or options.target.name != '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':
        raise ValueError('Pinned native target implementation and checkpoint required')
    admission['target_sources'] = actual
    from t32_hardware_kernel import installed
    with installed(root, options.proposal_evidence, directory, fused_score_evidence=options.score_evidence) as installation:
        from t32_score_hardware import native_sources, sources as score_sources
        admission['installed_native_sources'] = native_sources(root)
        admission['score_sources'] = score_sources(directory)
        admission['runtime_preflight'] = installation
    if options.preflight:
        options.output.write_text(json.dumps(admission, indent=2))
        return
    run_request(options, load_rotary(options.config), admission)


def run_request(options, rotary, admission):
    require_projection_environment(os.environ, True)
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
    from models.tt_transformers.tt.ccl import TT_CCL
    from coding_context_request import make_context_prompt
    from t32_combined_experiment import run_loaded_requests

    report = dict(passed=False, closed_cleanly=False,
        scope=__doc__, backend='hardware',
        source_revision=os.environ.get('QWEN_SOURCE_REVISION'), workflow_run=os.environ.get('QWEN_WORKFLOW_RUN'),
        streams=1, context=4096, proposals=31, pp=None, committed_tg=None,
        hardware_qualified=False, attention=admission, sources=sources(), resources_before=snapshot())
    owned, mesh = [], None
    started = time.perf_counter()
    report['stages'] = []

    def progress(stage):
        report['stage'] = stage
        entry = dict(stage=stage, elapsed_seconds=time.perf_counter() - started)
        report['stages'].append(entry)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(entry), flush=True)

    try:
        os.environ['MODEL_WEIGHTS_DIR'] = str(options.target)
        os.environ['HF_MODEL'] = str(options.target)
        config = AutoConfig.from_pretrained(options.target, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(options.target, local_files_only=True, trust_remote_code=False)
        prompt, context = make_context_prompt(tokenizer, context_tokens=4096)
        progress('open_mesh')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1073741824)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        progress('load_complete_target')
        generator = Qwen36ForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=8, max_seq_len=65536)
        model = generator.model[0]
        if len(model.layers) != 64 or model.vocab_size != 248320 or model._lmhead_vocab_sharded is not True:
            raise ValueError('Complete TP2 target required')
        kv_cache = generator.allocate_kv_cache((1032, model.args.n_local_kv_heads, 64, model.args.head_dim), ttnn.bfloat16, 64)
        pages = torch.arange(1024, dtype=torch.int32).reshape(1, 1024)

        def upload(value, sharded=False, row_major=False):
            tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
                if sharded else ttnn.ReplicateTensorToMesh(mesh))
            owned.append(tensor)
            return tensor

        parameters = {}
        with VerifiedWeights(options.checkpoint) as reader:
            report['draft_weight_hashes'] = reader.fingerprints()
            for name in PARAMETERS:
                progress('load_' + name)
                value, sharded = pack_parameter(name, reader.tensor(name))
                parameters[name] = upload(value, sharded)
                del value
            predecessor = upload(reader.tensor('markov_head.markov_w1.weight').reshape(1, 1, 248320, 256), row_major=True)
            successor = upload(reader.tensor('markov_head.markov_w2.weight').T.contiguous().reshape(1, 1, 256, 248320))
        layer_weights = [{name: parameters[f'layers.{index}.{name}'] for name in SPECIFICATIONS} for index in range(5)]
        run_loaded_requests(ttnn, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
            layer_weights, predecessor, successor, rotary, report, progress,
            prompt=prompt, context=context, proposal_evidence=options.proposal_evidence,
            score_evidence=options.score_evidence, attention_evidence=options.attention_evidence)
        progress('complete')
        report['passed'] = True
    except BaseException as error:
        report.update(passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
            report['sources_after'] = sources()
            report['resources_after'] = snapshot()
            if report['sources'] != report['sources_after']:
                raise ValueError('Sources or active runtime changed during request')
        except BaseException as error:
            report.update(passed=False, cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
