"""Full-model B1 host-sampling trace attribution through the served Generator."""

import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from tracy import signpost
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("baseline-client.py"))
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    root = Path("/experiment/results/model-profile")
    report = dict(passed=False, scope="Full-model host-sampling Generator trace; excludes vLLM scheduler/HTTP; instrumented timing only")
    mesh = None
    generator = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1073741824)
        mesh.enable_program_cache()
        weights = os.environ["MODEL_WEIGHTS_DIR"]
        config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
        prompt = baseline.make_prompt(tokenizer, 128, 0)
        generator = Qwen36ForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=8, max_seq_len=65536)
        model = generator.model[0]
        kv_cache = generator.allocate_kv_cache((8200, model.args.n_local_kv_heads, 64, model.args.head_dim), ttnn.bfloat16, len(model.layers))
        report.update(layers=len(model.layers), max_batch=8, active_batch=1, context=65536, kv_blocks=8200,
                      prompt_tokens=len(prompt), vocab=model.args.vocab_size)
        if len(prompt) != 109 or model.args.vocab_size != 248320:
            raise AssertionError("Unexpected pinned model/prompt")
        ttnn.ReadDeviceProfiler(mesh)
        generator.warmup_model_prefill(kv_cache=kv_cache, enable_trace=True)
        ttnn.ReadDeviceProfiler(mesh)
        for trace in (False, True):
            generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=trace, max_batch_size=1,
                                         num_blocks=1024, can_sample_on_device=False)
            ttnn.ReadDeviceProfiler(mesh)
        page_table = torch.arange(1024, dtype=torch.int32).reshape(1, 1024)
        logits, _ = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), page_table,
                                               kv_cache, [len(prompt)], empty_slots=[0], enable_trace=True)
        generated = [int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax())]
        ttnn.ReadDeviceProfiler(mesh)
        report["steps"] = []
        print("QWEN_PROFILE_MEASURE_BEGIN", flush=True)
        for step in range(15):
            signpost(f"qwen_profile_decode_{step}")
            started = time.perf_counter()
            result = generator.decode_forward(tokens=torch.tensor([[generated[-1]]], dtype=torch.int32),
                start_pos=torch.tensor([len(prompt) + step], dtype=torch.int32), page_table=page_table,
                kv_cache=kv_cache, enable_trace=True, read_from_device=True)
            logits = result[0] if isinstance(result, tuple) else result
            generated.append(int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax()))
            report["steps"].append(dict(step=step, instrumented_host_seconds=time.perf_counter() - started))
            ttnn.ReadDeviceProfiler(mesh)
        print("QWEN_PROFILE_MEASURE_END", flush=True)
        report["decode_trace_id"] = int(generator.trace_ids_decode[False][0])
        report["token_ids"] = generated
        expected = [1596, 1144, 4087, 1156, 25, 328, 727, 18054, 74830, 11, 1328, 1590, 198, 262, 460, 1727]
        report["reference_run"] = 33950377324
        if generated != expected:
            raise AssertionError("Profiled Generator differs from the endpoint control's first 16 tokens")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "generation.json").write_text(json.dumps(report, indent=2))
        if mesh is not None:
            ttnn.ReadDeviceProfiler(mesh)
            if generator is not None:
                del generator
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
