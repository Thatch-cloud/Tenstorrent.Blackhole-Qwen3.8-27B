"""Independent completeness checks for the synthetic T32 Markov simulator report."""


def validate(report, *, learned=False):
    from dspark_markov_fixture import expected_manifest

    if type(learned) is not bool or (learned and report.get('fixture') != expected_manifest()):
        raise ValueError('Explicit learned policy and exact pinned matrix manifest required')
    patterns = 2 if learned else 3
    replay_patterns = (*range(patterns), 0)
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('proposals') != 31 or report.get('vocabulary') != (248320 if learned else 64)
            or report.get('native_arithmetic_reference') is not True
            or report.get('target_integrated') is not False or report.get('eligible_for_hardware') is not False
            or not report.get('sources') or report['sources'] != report.get('sources_after')
            or not report.get('native_sources') or report['native_sources'] != report.get('native_sources_after')):
        raise ValueError('Complete unchanged synthetic T32 simulator report required')

    def coverage(field, keys, expected, flags):
        records = report.get(field, [])
        actual = [tuple(record.get(key) for key in keys) for record in records]
        if len(actual) != len(expected) or set(actual) != expected or any(
                record.get(flag) is not True for record in records for flag in flags):
            raise ValueError('Missing, duplicate or failed T32 coverage: ' + field)

    coverage('eager_checks', ('pattern', 'chip', 'step'),
        {(pattern, chip, step) for pattern in range(patterns) for chip in range(2) for step in range(31)},
        ('token_exact', 'full_vocabulary_exact'))
    coverage('replay_checks', ('repetition', 'pattern', 'chip', 'step'),
        {(repetition, pattern, chip, step) for repetition, pattern in enumerate(replay_patterns)
            for chip in range(2) for step in range(31)}, ('token_and_scores_exact', 'bindings_stable'))
    coverage('input_checks', ('phase', 'ordinal', 'tensor', 'chip'),
        {(phase, ordinal, tensor, chip) for phase, count in (('eager', patterns), ('replay', patterns + 1))
            for ordinal in range(count) for tensor in range(2) for chip in range(2)}, ('exact',))
    coverage('weight_checks', ('phase', 'tensor', 'chip'),
        {(phase, tensor, chip) for phase in ('before', 'after') for tensor in range(2) for chip in range(2)},
        ('exact',))
    coverage('stale_controls', ('chip',), {(0,), (1,)}, ('missing_update_detected',))
    return dict(passed=True, eager_queries=patterns * 62, replay_queries=(patterns + 1) * 62,
        target_integrated=False, learned=learned,
        scope=('Learned full-vocabulary matrices with synthetic base logits' if learned else 'Synthetic 64-token vocabulary')
            + '; no throughput or full-model qualification')
