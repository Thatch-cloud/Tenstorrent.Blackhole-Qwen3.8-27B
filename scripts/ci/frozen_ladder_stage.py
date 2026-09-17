"""Stage identical offline combined sources for every requested context geometry."""

import argparse
import hashlib
import json
from pathlib import Path

CHANGES = {
    'frozen_combined_history.py': (
        ('history_limit() != 33024', "history_limit() != selected_geometry()['capacity']"),
        ('position != 32768', "position != selected_geometry()['context']")),
    'frozen_draft_tail_scope.py': (
        ("options.get('position') != 33024", "options.get('position') != selected_geometry()['capacity']"),),
    'frozen_incremental_scope.py': (
        ("record.get('initial_position') != 32768 or record.get('capacity') != 33024",
            "record.get('initial_position') != selected_geometry()['context'] or record.get('capacity') != selected_geometry()['capacity']"),
        ("os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '32768'",
            "os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != str(selected_geometry()['context'])")),
    'frozen_combined_runtime.py': (
        ('position != 32768', "position != selected_geometry()['context']"),
        ("            or selected_geometry()['context'] != 32768\n", ''),
        ("result.update(report_sha256=REPORTS['draft-numerical.json'], runtime_component_sources=expected)",
            "result.update(report_sha256=REPORTS['draft-numerical.json'], runtime_component_sources=expected,\n"
            "        component_reference_context=32768, requested_geometry=selected_geometry(),\n"
            "        requested_geometry_qualified=False, requires_fresh_full_request_audit=True)")),
    'dspark-target-hardware.py': (
        ('request_context() == 32768', "request_context() == selected_geometry()['context']"),
        ('history_limit() != 33024', "history_limit() != selected_geometry()['capacity']")),
    'dspark_runtime_cache.py': (
        ('enabled=request_context() == 32768', "enabled=request_context() == selected_geometry()['context']"),
        ("inputs['draft_8k_factory'] = factory",
            "from frozen_ladder_cache import build_identity\n"
            "        inputs['draft_8k_factory'] = build_identity(factory)")),
    'dspark_8k_build.py': (
        ('capacity=33024, target_tree_scratch=scratch', "capacity=selected_geometry()['capacity'], target_tree_scratch=scratch"),),
    'target_t16_attention_gate.py': (
        ('request_context() == 32768', "request_context() == selected_geometry()['context']"),),
    'dspark_8k_entry.py': (
        ('request_context() != 32768', "request_context() != selected_geometry()['context']"),
        ('context=32768, output_tokens=256', "context=selected_geometry()['context'], output_tokens=256")),
    'dspark_8k_admission.py': (
        ('context != 32768', "context != selected_geometry()['context']"),
        ('capacity=33024, key_chunk_size=256', "capacity=selected_geometry()['capacity'], key_chunk_size=256")),
    'dspark_context_selection.py': (
        ('history_limit() == 33024', "history_limit() == selected_geometry()['capacity']"),),
    'dspark_request_experiment.py': (
        ("            report['request_checks'].append(result)",
            "            if audit:\n"
            "                from frozen_ladder_requests import validate_audit\n"
            "                validate_audit(result)\n"
            "            report['request_checks'].append(result)"),
        ('num_blocks=1024, can_sample_on_device=False', "num_blocks=selected_geometry()['target_page_count'], can_sample_on_device=False"),
        ('not 1 <= valid <= 65536', "not 1 <= valid <= selected_geometry()['target_sequence_capacity']"),
        ("    report['coding_context'], report['request_checks'] = context, []",
            "    if profile_verifier or profile_drafter or combined_profile:\n"
            "        raise ValueError('Unprofiled combined ladder requests required')\n"
            "    schedule = (('publication', True), ('publication', False), ('publication', False))\n"
            "    report['coding_context'], report['request_checks'] = context, []"),
        ),
}


def adapt(sources):
    result = dict(sources)
    for name, changes in CHANGES.items():
        source = result[name]
        for before, after in changes:
            count = source.count(before)
            expected = 2 if (name in ('dspark-target-hardware.py', 'target_t16_attention_gate.py')
                and before == 'request_context() == 32768') else 1
            if count != expected:
                raise ValueError(f'{name}: expected {expected} occurrences of {before!r}, found {count}')
            source = source.replace(before, after)
        if 'from frozen_context_geometry import selected_geometry' not in source:
            source = 'from frozen_context_geometry import selected_geometry\n' + source
        if name == 'dspark_request_experiment.py':
            anchor = '    if profile_verifier or profile_drafter or combined_profile:\n        report.update('
            if source.count(anchor) != 1:
                raise ValueError('Original summary boundary required')
            source = source.split(anchor)[0] + '    from frozen_ladder_requests import finish\n    finish(report, summarize)\n'
        compile(source, name, 'exec')
        result[name] = source
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument('--wide-cache-evidence', type=Path)
    cache_group.add_argument('--full-window-cache-evidence', type=Path)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh ladder staging required')
    scripts = options.checkout / 'scripts/ci'
    if (scripts / 'frozen-request-phase.json').exists():
        raise ValueError('Ladder requires fresh audits, not retained phase qualification')
    before = {name: (scripts / name).read_bytes() for name in CHANGES}
    result = adapt({name: payload.decode() for name, payload in before.items()})
    result['coding_context_request.py'] = 'from frozen_ladder_prompt import make_context_prompt\n'
    for name in ('frozen_ladder_prompt.py', 'frozen_ladder_requests.py', 'frozen_ladder_cache.py'):
        result[name] = Path(__file__).with_name(name).read_text()
    payloads = {name: source.encode() for name, source in result.items()}
    payloads['frozen-ladder-corpus.json'] = Path(__file__).with_name('frozen-ladder-corpus.json').read_bytes()
    if options.wide_cache_evidence:
        payloads.update(wide_cache_payloads(scripts, options.wide_cache_evidence,
            payloads['dspark_request_experiment.py']))
    if options.full_window_cache_evidence:
        payloads.update(wide_cache_payloads(scripts, options.full_window_cache_evidence,
            payloads['dspark_request_experiment.py'], full_window=True))
    for name, payload in payloads.items():
        (scripts / name).parent.mkdir(parents=True, exist_ok=True)
        (scripts / name).write_bytes(payload)
    from frozen_draft_tail_gate import qualify_hardware
    qualify_hardware(scripts, scripts / 'frozen-draft-tail-hardware.json')
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(payload).hexdigest() for name, payload in before.items()},
        after={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        component_reference_context=32768, geometry_qualified=False,
        requires_fresh_full_request_audit=True, performance_qualified=False,
        serving_defaults_changed=False), indent=2) + '\n')


def wide_cache_payloads(scripts, evidence, request_source, *, full_window=False):
    from frozen_ladder_cache_gate import qualify
    evidence = Path(evidence)
    raw = (evidence / 'reports.json').read_bytes()
    digests = json.loads(raw)
    if type(full_window) is not bool:
        raise ValueError('Explicit full-window selection required')
    expected = {'261888'} if full_window else {'65536', '131072'}
    if set(digests) != expected:
        raise ValueError('Full-window simulator report required' if full_window
            else 'Both long-context simulator reports required')
    result = {}
    for name in ('frozen_ladder_cache_scope.py', 'frozen_ladder_cache_gate.py',
            'frozen_ladder_ordered_cache.py', 'frozen_ladder_reference_memory.py', 'ladder-cache-probe.py'):
        result[name] = Path(__file__).with_name(name).read_bytes()
    for context, digest in digests.items():
        report = qualify(Path(__file__).parent, evidence / context, digest, int(context))
        for name in ('frozen_context_geometry.py', 'ordered_cache.py', 'attention_batch.py'):
            if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != report['sources'][name]:
                raise ValueError('Staged cache source differs from simulator: ' + name)
        for name in ('ladder-cache.json', 'ladder-cache.exit-status', 'simulator-runtime.txt'):
            result[f'frozen-cache-evidence/{context}/{name}'] = (evidence / context / name).read_bytes()
    result['frozen-cache-evidence/reports.json'] = raw
    source = request_source.decode()
    anchor = '    with sampler_links(sampler.tt_sampling, 4):\n'
    if source.count(anchor) != 1:
        raise ValueError('Unchanged combined request scope required')
    source = source.replace(anchor,
        '    from frozen_ladder_cache_scope import runtime_scope as cache_scope\n'
        '    with cache_scope(Path(__file__).parent) as cache_evidence, sampler_links(sampler.tt_sampling, 4):\n'
        "        report['wide_cache_geometry'] = cache_evidence\n")
    compile(source, 'dspark_request_experiment.py', 'exec')
    result['dspark_request_experiment.py'] = source.encode()
    return result


if __name__ == '__main__':
    main()
