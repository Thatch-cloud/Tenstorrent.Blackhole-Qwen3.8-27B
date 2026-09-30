"""Allow bounded timing only after this mesh completes a fresh full-request audit."""

from dspark_intake import TAPS


def validate(request, native_checks):
    dspark = request.get('dspark', {})
    fusion = request.get('fused_t32_mlp', {})
    blocks = request.get('blocks', [])
    if (any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact', 'instrumented_timing'))
            or request.get('length') != 4096 or not blocks
            or dspark.get('proposals') != 31 or dspark.get('verifier_rows') != 32
            or dspark.get('t32_lifecycle', {}).get('hardware_audit_experiment') is not True
            or request.get('committed_tokens_per_second') is not None
            or fusion.get('rows') != 32 or len(fusion.get('hits', [])) != 64
            or any(type(value) is not int or value <= 0 for value in fusion['hits'])
            or any(fusion.get(key) is not True for key in ('restored', 'native_bindings_unchanged'))):
        raise ValueError('Fresh exact combined T32 audit with every fused target layer required')
    positions = [4096, *(block['position'] for block in blocks if block['rows'] > 1)]
    for checks in (native_checks, dspark.get('proposal_checks', [])):
        if ([entry.get('position') for entry in checks] != positions
                or any(entry.get('exact') is not True or entry.get('tensors') != 6 for entry in checks)):
            raise ValueError('Every complete native-score and changed-input replay comparison required')
    expected = {(block['position'], block['committed'], tap, chip)
        for block in blocks for tap in TAPS for chip in (0, 1)}
    features = dspark.get('feature_checks', [])
    if (len(features) != len(expected) or {(entry.get('position'), entry.get('rows'), entry.get('tap'), entry.get('chip'))
            for entry in features} != expected or any(entry.get('exact') is not True for entry in features)):
        raise ValueError('All committed features on both chips required before timing')
    expected_gdn = [dict(position=block['position'], rows=block['rows'], unchanged=True)
        for block in blocks if block['rows'] > 1]
    if request.get('gdn_verify_checks') != expected_gdn:
        raise ValueError('Every speculative block must leave native recurrent state unchanged before commit')
    return dict(prompt_tokens=tuple(request['prompt_tokens']), emitted=tuple(request['emitted']),
        max_new_tokens=request['max_new_tokens'])


def authorize(request):
    from t32_score_hardware import require_active

    state = require_active()
    if state.timing_reference is not None:
        raise ValueError('Only one fresh audit may authorize timing in this scope')
    state.timing_reference = validate(request, state.record['native_proposal_checks'])
    state.record['fresh_request_audit_passed'] = True


def validate_timed(request):
    from t32_score_hardware import require_active

    reference = require_active().timing_reference
    if (reference is None or request.get('instrumented_timing') is not False
            or any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact'))
            or any(tuple(request[key]) != reference[key] for key in ('prompt_tokens', 'emitted'))
            or request.get('max_new_tokens') != reference['max_new_tokens']):
        raise ValueError('Timed request must reproduce its fresh audited prompt, output and state')
