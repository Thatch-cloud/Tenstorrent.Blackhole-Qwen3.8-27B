"""Owned complete-model T32 hardware audit after target and draft weights are loaded."""

import os
from pathlib import Path

from t32_hardware_kernel import installed, request_admission
from t32_score_hardware import hardware_scope


def run_loaded_requests(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
        layer_weights, predecessor, successor, rotary, report, progress, *, prompt, context,
        proposal_evidence, score_evidence, attention_evidence):
    from dspark_request_experiment import run_loaded_requests as run

    if len(prompt) != 4096 or report.get('streams') != 1:
        raise ValueError('Single-stream 4K complete combined audit required')
    directory = Path(__file__).parent
    root = Path(os.environ['TT_METAL_HOME'])
    with installed(root, proposal_evidence, directory, fused_score_evidence=score_evidence) as installation:
        report['t32_hardware_installation'] = installation
        with request_admission(root, installation), hardware_scope(root, directory, proposal_evidence,
                score_evidence, model.mesh_device, installation) as audit:
            report['t32_score_hardware_audit'] = audit
            run(operations, generator, model, collectives, tokenizer, pages, kv_cache, parameters,
                layer_weights, predecessor, successor, rotary, report, progress, prompt=prompt, context=context,
                max_new_tokens=65, t32_attention_evidence=attention_evidence)
            if not audit['native_proposal_checks']:
                raise ValueError('Complete learned native-score proposal comparison required')
