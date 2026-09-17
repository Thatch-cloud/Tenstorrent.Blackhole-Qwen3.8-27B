"""One fresh full-model feature audit and two complete timed requests per context."""

from frozen_context_geometry import selected_geometry


def validate_audit(audit):
    from gdn_shared_qk_variants import validate_route
    validate_route(audit, 'publication')
    if (audit.get('instrumented_timing') is not True or audit.get('length') != selected_geometry()['context']
            or any(audit.get(field) is not True for field in ('exact', 'state_exact', 'inactive_exact'))):
        raise ValueError('Fresh full-context exact output and state audit required')
    blocks = [block for block in audit['blocks'] if block['rows'] > 1]
    checks = audit.get('dspark', {}).get('proposal_checks', [])
    if (not blocks or [check.get('position') for check in checks]
            != [audit['length'], *(block['position'] for block in blocks)]
            or any(check.get('exact') is not True or check.get('tensors') != 6 for check in checks)):
        raise ValueError('Complete eager versus replay feature audit required')
    if audit.get('gdn_verify_checks') != [dict(position=block['position'], rows=block['rows'], unchanged=True)
            for block in blocks]:
        raise ValueError('Complete deferred-publication state audit required')


def finish(report, summarize):
    from dspark_request_variants import proposal_signature
    from gdn_shared_qk_variants import validate_route
    requests = report['request_checks']
    if [(value.get('arm'), value.get('instrumented_timing')) for value in requests] != [
            ('publication', True), ('publication', False), ('publication', False)]:
        raise ValueError('Fresh candidate audit followed by two timed requests required')
    audit = requests[0]
    validate_audit(audit)
    for value in requests[1:]:
        validate_route(value, 'publication')
        if value.get('length') != selected_geometry()['context']:
            raise ValueError('Full requested prompt length required')
        if proposal_signature(value) != proposal_signature(audit):
            raise ValueError('Timed proposals must exactly reproduce the fresh feature audit')
    summary = summarize(requests)
    report.update(request_summary=summary, ctx_tokens=summary['ctx'], pp=summary['pp'],
        committed_tg=summary['committed_tg'], drafter_history_rows=summary['ctx'], proposal_rows=15,
        comparison_axis='Same combined recipe; context is the only runtime setting varied',
        fresh_context_audit=True, scope=__doc__)
