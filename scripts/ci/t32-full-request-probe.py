"""Actual T32 coding request in the simulator; no hardware throughput qualification."""

import argparse
import json
import os
from pathlib import Path

from dspark_hardware_gate import digest
from dspark_intake import FILES
from dspark_pipeline import PARAMETERS, pack_parameter
from dspark_layer import SPECIFICATIONS
from dspark_rope_tables import DSparkRotary
from dspark_weights import VerifiedWeights
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from native_draft_sdpa import run_precise_probe
from sim_memory_budget import require_clean
from t32_attention_admission import require_active
from t32_ci_runtime import snapshot


def sources():
    roots = (Path(__file__).parent, Path('/speculative-decoding/harness'))
    return {str(path): digest(path) for root in roots for path in sorted(root.rglob('*'))
        if path.suffix in ('.py', '.cpp', '.sh')}


def load_rotary(path):
    return DSparkRotary(json.loads(path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'target', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))):
        raise ValueError('T32 request probe is simulator-only, without allocated cards')
    require_projection_environment(os.environ, False)
    if options.output.exists() or digest(options.config) != FILES['config.json'][1]:
        raise ValueError('Fresh output and pinned draft configuration required')
    rotary = load_rotary(options.config)
    run_precise_probe(__file__)
    admission = require_active()
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
    from models.tt_transformers.tt.ccl import TT_CCL
    from coding_context_request import make_context_prompt
    from dspark_request_experiment import run_loaded_requests

    report = dict(passed=False, closed_cleanly=False, scope=__doc__, backend='simulator',
        streams=1, context=4096, proposals=31, pp=None, committed_tg=None,
        hardware_qualified=False, attention=admission, sources=sources(), resources_before=snapshot())
    owned, mesh = [], None

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)

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
            prompt=prompt, context=context, t32_request=True)
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
            require_clean(report['resources_before'], report['resources_after'])
            if report['sources'] != report['sources_after'] or require_active() != admission:
                raise ValueError('Sources or active runtime changed during request')
        except BaseException as error:
            report.update(passed=False, cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
