"""Value-only synthetic probes to isolate normalization and influential key probabilities."""


KINDS = ('constant', 'oldest', 'last_proposal')


def diagnostic_fixture(fixture, kind, position, proposals):
    if kind not in KINDS:
        raise ValueError('Unknown value diagnostic')
    values = dict(fixture)
    for name in ('history_value', 'query_value'):
        values[name] = fixture[name].clone()
    values['history_value'][:, :, :position] = 1 if kind == 'constant' else 0
    values['query_value'][:, :, :proposals] = 1 if kind == 'constant' else 0
    if kind == 'oldest':
        values['history_value'][:, :, 0] = 1
    elif kind == 'last_proposal':
        values['query_value'][:, :, proposals - 1] = 1
    return values
