"""Conservative staging policy for the experimental folded-attention fixture."""


def requires_singletons(fixture, enabled):
    if enabled not in ('0', '1'):
        raise ValueError('Explicit singleton staging experiment switch required')
    if enabled == '0':
        return True
    from attention_batch import OrderedCacheWriter
    from attention_replay import ReplayAttentionReader

    reader = getattr(fixture, 'replay_reader', None)
    writers = getattr(fixture, 'writers', ())
    readers = getattr(fixture, 'readers', ())
    return not (type(reader) is ReplayAttentionReader and reader.audit is None
        and getattr(fixture, 'ordered_cache', False) is True
        and len(writers) == 16 and all(type(writer) is OrderedCacheWriter for writer in writers)
        and len(readers) == 16 and all(value is reader for value in readers))
