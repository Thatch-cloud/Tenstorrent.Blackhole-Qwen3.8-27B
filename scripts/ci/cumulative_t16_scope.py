"""Compose qualified target windows and draft selection within one request."""

from contextlib import contextmanager

from compact_score_scope import scoped_compact_scores
from gdn_direct_window_scope import scoped_direct_windows


@contextmanager
def scoped_cumulative_t16(direct_admission, compact_admission, directory):
    with scoped_direct_windows(direct_admission, directory) as direct:
        with scoped_compact_scores(compact_admission, directory) as compact:
            yield dict(direct=direct, compact=compact)
    if direct.get('restored') is not True or compact.get('restored') is not True:
        raise ValueError('Both cumulative routes must restore their native bindings')


def validate_request(request, audit):
    direct, compact = audit['direct'], audit['compact']
    if (direct.get('restored') is not True or compact.get('restored') is not True
            or type(direct.get('hits')) is not int or direct['hits'] < 48 or direct['hits'] % 48
            or direct['hits'] != len(request.get('gdn_shared_qk', {}).get('loads', []))
            or type(compact.get('calls')) is not int or compact['calls'] < 1
            or compact['calls'] != request.get('score_layout', {}).get('calls')
            or compact.get('steps') != 15 * compact['calls']):
        raise ValueError('Both cumulative components must execute and restore in the same request')
