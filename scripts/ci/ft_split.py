"""WP-F1: the split guard of the drafter fine-tune's data (Python 3.7, stdlib).

The fine-tune must never train on what it is judged on. This module REFUSES (it does not warn) when a training set overlaps a held-out
set, and it carries the two standing data rules:
  * SWE turns are split by REPOSITORY, and the training repositories are disjoint from A1 (the screen's held-out SWE turns) and from the
    1,000-turn eval tier;
  * own traces are disjoint from A2 by conversation AND by project, and they enter ONLY after an explicit owner decision (D4):
    until `d4_cleared` is passed they are refused outright.
The default mix is the D4 fallback, 50 / 30 / 20 SWE / chained / code tasks, which needs no own trace.

Records are dicts with `source` (swe / chained / code / own) and the identifiers that source needs: `repo` (swe), `group` (chained,
code), `conversation` and `project` (own). Messages name an identifier's KIND and a COUNT, never its value.
"""
import hashlib

DEFAULT_MIX = (('swe', 0.5), ('chained', 0.3), ('code', 0.2))
SOURCES = ('swe', 'chained', 'code', 'own')
KEYS = {'swe': ('repo',), 'chained': ('group',), 'code': ('group',), 'own': ('conversation', 'project')}


class SplitError(ValueError):
    """A split the guard refuses."""


def check_mix(mix, d4_cleared=False):
    """A mix [(source, share)] must name known sources, sum to 1, and name `own` only when D4 is cleared."""
    mix = list(mix)
    names = [name for name, _ in mix]
    if any(name not in SOURCES for name in names) or len(set(names)) != len(names):
        raise SplitError('the mix names an unknown or repeated source')
    if any(share <= 0 for _, share in mix) or abs(sum(share for _, share in mix) - 1.0) > 1e-9:
        raise SplitError('the mix shares must be positive and sum to 1')
    if 'own' in names and not d4_cleared:
        raise SplitError('own traces need the provider-terms decision (D4) first')
    return mix


def mix_counts(total, mix):
    """Whole example counts per source summing to `total` (largest remainder)."""
    base = dict((name, int(total * share)) for name, share in mix)
    left = total - sum(base.values())
    for name, share in sorted(mix, key=lambda item: (-(total * item[1] - base[item[0]]), item[0]))[:left]:
        base[name] += 1
    return base


def split_by_key(records, key, held_out_fraction, seed):
    """Deterministic split by the value of `key`: every record of one value lands on one side. -> (train, held_out)."""
    if not 0.0 < held_out_fraction < 1.0:
        raise SplitError('the held-out fraction must be between 0 and 1')
    train, held = [], []
    for record in records:
        if key not in record:
            raise SplitError('a record has no %s' % key)
        digest = hashlib.sha256(('%s|%s' % (seed, record[key])).encode('utf-8')).digest()
        (held if int.from_bytes(digest[:8], 'big') / float(1 << 64) < held_out_fraction else train).append(record)
    return train, held


def values_of(records, key):
    return set(record[key] for record in records if key in record)


def assert_disjoint(train, held_out, keys, what='held-out set'):
    """Refuse when any value of any key is in both. The message carries the key and the count."""
    for key in keys:
        overlap = values_of(train, key) & values_of(held_out, key)
        if overlap:
            raise SplitError('%d %s values are in both the training set and the %s' % (len(overlap), key, what))


def guard_training_set(train, a1=(), a2=(), eval_tier=(), d4_cleared=False):
    """Every standing rule over a training set (records as described above):
      swe: `repo` disjoint from A1 and the eval tier;  own: refused without D4, else `conversation` and `project` disjoint from A2;
      chained / code: `group` disjoint from the eval tier.  Every record's source and identifiers must be present."""
    for record in train:
        source = record.get('source')
        if source not in SOURCES:
            raise SplitError('a record has no known source')
        for key in KEYS[source]:
            if key not in record:
                raise SplitError('a %s record has no %s' % (source, key))
    if any(record['source'] == 'own' for record in train) and not d4_cleared:
        raise SplitError('own traces need the provider-terms decision (D4) first')
    swe = [record for record in train if record['source'] == 'swe']
    assert_disjoint(swe, [r for r in a1 if r.get('source') == 'swe' or 'repo' in r], ['repo'], 'A1 turns')
    assert_disjoint(swe, [r for r in eval_tier if 'repo' in r], ['repo'], 'eval tier')
    own = [record for record in train if record['source'] == 'own']
    assert_disjoint(own, a2, ['conversation', 'project'], 'A2 sessions')
    grouped = [record for record in train if record['source'] in ('chained', 'code')]
    assert_disjoint(grouped, [r for r in eval_tier if 'group' in r], ['group'], 'eval tier')
    return True
