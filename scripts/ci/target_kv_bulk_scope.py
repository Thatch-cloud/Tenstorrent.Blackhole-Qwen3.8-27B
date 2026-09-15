"""Audited-only admission for fewer KV transfers, with the old digest as oracle."""

from time import perf_counter

from target_kv_bulk_audit import digest_prefix


def qualified_callback(original, candidate, records, emit, *, clock=perf_counter):
    selected = None

    def timed(operation, valid):
        started = clock()
        result = operation(valid)
        return result, (clock() - started) * 1000

    def invoke(valid):
        nonlocal selected
        if selected is None:
            if type(valid) is not int or valid != 65536:
                raise ValueError('First bulk KV admission must cover the full 64K frontier')
            comparisons = []
            for prefix in (valid, valid - 1):
                expected, reference_ms = timed(original, prefix)
                actual, candidate_ms = timed(candidate, prefix)
                exact = actual == expected
                record = dict(event='bulk_kv_admission', valid=prefix, exact=exact,
                    reference_ms=reference_ms, candidate_ms=candidate_ms,
                    performance_qualified=False)
                records.append(record)
                emit(record)
                if not exact:
                    raise AssertionError('Bulk target KV digest differs from legacy oracle')
                comparisons.append((reference_ms, candidate_ms))
                if prefix == valid:
                    result = actual
            selected = candidate if sum(value[1] for value in comparisons) < sum(
                value[0] for value in comparisons) else original
            emit(dict(event='bulk_kv_selected', candidate=selected is candidate,
                performance_qualified=False))
            return result
        result, elapsed_ms = timed(selected, valid)
        record = dict(event='bulk_kv_snapshot', valid=valid, elapsed_ms=elapsed_ms,
            candidate=selected is candidate, performance_qualified=False)
        records.append(record)
        emit(record)
        return result

    return invoke


def install_callback(arguments, records, emit):
    from dspark_64k_scope import validate_target_kv_prefix
    from dspark_projection import tensor_digest
    from gdn_multitoken_conv import addresses

    if arguments.get('audit_features') is not True:
        raise ValueError('Bulk KV admission is audit-only')
    operations = arguments['operations']
    caches = [value for pair in arguments['model']._paged_kv_caches for value in pair]
    if len(caches) != 32:
        raise ValueError('All 32 target KV allocations required')

    def candidate(valid):
        validate_target_kv_prefix(valid, caches)
        statistics = {}
        result = digest_prefix(operations, caches, valid, digest=tensor_digest,
            addresses=addresses, evidence=statistics)
        emit(dict(event='bulk_kv_readback', **statistics, performance_qualified=False))
        return result

    arguments['kv_digest'] = qualified_callback(arguments['kv_digest'], candidate, records, emit)
