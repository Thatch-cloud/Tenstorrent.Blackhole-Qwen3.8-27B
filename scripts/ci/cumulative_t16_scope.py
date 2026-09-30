"""Compose qualified target windows and draft selection within one request."""

from contextlib import contextmanager, ExitStack

from compact_score_scope import scoped_compact_scores
from gdn_direct_window_scope import scoped_direct_windows
from mlp_down_grid_scope import scoped_down_grid


@contextmanager
def scoped_cumulative_t16(direct_admission, compact_admission, directory, *, down_admission=None,
                         drafter='dspark'):
    if drafter not in ('dspark', 'dflash2'):
        raise ValueError('Explicit DSpark or DFlash2 drafter required')
    if (drafter == 'dspark') != (compact_admission is not None):
        raise ValueError('Compact Markov admission belongs only to DSpark')
    with ExitStack() as stack:
        audit = dict(direct=stack.enter_context(scoped_direct_windows(direct_admission, directory)))
        if drafter == 'dspark':
            audit['compact'] = stack.enter_context(scoped_compact_scores(compact_admission, directory))
        if down_admission is not None:
            audit['down'] = stack.enter_context(scoped_down_grid(down_admission))
        yield audit
    if any(component.get('restored') is not True for component in audit.values()):
        raise ValueError('All cumulative routes must restore their native bindings')


def validate_request(request, audit, *, drafter='dspark'):
    if drafter not in ('dspark', 'dflash2'):
        raise ValueError('Explicit DSpark or DFlash2 drafter required')
    if drafter == 'dflash2' and (request.get('selected_drafter') != 'dflash2' or 'compact' in audit):
        raise ValueError('DFlash2 must execute without the DSpark compact selector')
    direct = audit['direct']
    if (direct.get('restored') is not True
            or type(direct.get('hits')) is not int or direct['hits'] < 48 or direct['hits'] % 48
            or direct['hits'] != len(request.get('gdn_shared_qk', {}).get('loads', []))):
        raise ValueError('Target direct windows must execute and restore in the same request')
    if drafter == 'dspark':
        compact = audit.get('compact', {})
        if (compact.get('restored') is not True
                or type(compact.get('calls')) is not int or compact['calls'] < 1
                or compact['calls'] != request.get('score_layout', {}).get('calls')
                or compact.get('steps') != 15 * compact['calls']):
            raise ValueError('Both cumulative components must execute and restore in the same request')
    if 'down' in audit:
        down = audit['down']
        hits = down.get('hits')
        if (down.get('restored') is not True or not isinstance(hits, list) or len(hits) != 64
                or any(type(count) is not int or count <= 0 for count in hits)
                or hits != request.get('fused_t16_mlp', {}).get('hits')):
            raise ValueError('Every fused MLP layer must execute the cumulative down-grid route')
