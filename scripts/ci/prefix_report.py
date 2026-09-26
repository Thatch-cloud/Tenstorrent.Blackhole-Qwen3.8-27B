"""The prefix-reuse timing report: TTFT, turn time, hit rate from the model's own markers (design 2.2
Timing checks). Pure: it reads the replay's records (prefix_replay) after prefix_judge.resolve.

Per phase (a number of busy agents) it records:
  - TTFT p50/p90 (seconds from sending a turn to its first generated token, queueing included) and
    turn time p50/p90 (to the last token: what an agent waits for), over every served turn and over
    continuation turns only (a first turn and a compaction are misses by construction);
  - the hit rate as the MODEL reports it: sum(Q) / sum(L) over the [PREFIX] rows of salted turns, and
    the share of continuation turns with Q > 0 - never vllm:prefix_cache_hits, which counts before the
    trim (kv_cache_manager.py:238-244);
  - the trim loss h - Q from the scheduler's grant lines, and the registry's orphan count when its
    stats are exported;
  - restore and capture milliseconds from the [PREFIX] rows, host RSS from docker stats, turns per
    hour, and the CI pod count the driver sampled (SKILL: CI shares the host).

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""


def percentile(values, fraction):
    """The nearest-rank percentile of `values` (None when empty); statistics.quantiles is 3.8+."""
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    rank = max(1, int(-(-fraction * len(values) // 1)))
    return values[min(len(values), rank) - 1]


def _round(value, digits=2):
    return None if value is None else round(value, digits)


def phase_summary(records, seconds=None, rss_gb=None, pods=None):
    """One phase's numbers from its records (each resolved by prefix_judge.resolve)."""
    served = [r for r in records if r.get('ok')]
    continuation = [r for r in served if r.get('continuation')]
    rows = [(r, (r.get('markers') or {})) for r in served if r.get('salt')]
    marked = [(r, m) for r, m in rows if m.get('q') is not None and m.get('l')]
    q_sum = sum(m['q'] for _, m in marked)
    l_sum = sum(m['l'] for _, m in marked)
    cont_marked = [(r, m) for r, m in marked if r.get('continuation')]
    grants = [m['grant'] for _, m in rows if m.get('grant')]
    restores = [row.get('restored_ms') for _, m in rows for row in m.get('rows') or () if row.get('q')]
    captures = [row.get('capture_ms') for _, m in rows for row in m.get('rows') or () if row.get('captured')]
    out = dict(
        turns=len(records), served=len(served), failed=len(records) - len(served),
        continuation_turns=len(continuation),
        ttft_p50=_round(percentile([r.get('ttft_s') for r in served], 0.5)),
        ttft_p90=_round(percentile([r.get('ttft_s') for r in served], 0.9)),
        ttft_continuation_p50=_round(percentile([r.get('ttft_s') for r in continuation], 0.5)),
        ttft_continuation_p90=_round(percentile([r.get('ttft_s') for r in continuation], 0.9)),
        turn_p50=_round(percentile([r.get('wall_s') for r in served], 0.5)),
        turn_p90=_round(percentile([r.get('wall_s') for r in served], 0.9)),
        prompt_tokens_mean=_round(sum(r.get('prompt_tokens') or 0 for r in served) / float(len(served)), 0)
        if served else None,
        answer_tokens_mean=_round(sum(r.get('completion_tokens') or 0 for r in served) / float(len(served)), 0)
        if served else None,
        hit_rate=_round(q_sum / float(l_sum), 4) if l_sum else None,
        hit_rate_continuation=_round(sum(m['q'] for _, m in cont_marked) / float(sum(m['l'] for _, m in cont_marked)), 4)
        if cont_marked else None,
        hit_turns=sum(1 for _, m in cont_marked if m['q']), marked_turns=len(marked),
        unmarked_salted_turns=len(rows) - len(marked),
        trim_loss_tokens=sum(max(0, g['h'] - g['q']) for g in grants), grants=len(grants),
        restore_ms_p50=_round(percentile(restores, 0.5)), restore_ms_p90=_round(percentile(restores, 0.9)),
        capture_ms_p50=_round(percentile(captures, 0.5)), capture_ms_p90=_round(percentile(captures, 0.9)),
        seconds=_round(seconds, 1), rss_gb_max=_round(rss_gb, 2), ci_pods=pods)
    if seconds:
        out['turns_per_hour'] = _round(len(served) * 3600.0 / seconds, 1)
    return out


def render_phase(name, summary):
    s = summary
    return ('%s: %s/%s turns served; TTFT p50/p90 %s/%s s (continuation %s/%s); turn p50/p90 %s/%s s; hit rate '
            '%s (continuation %s, %s of %s turns hit); trim loss %s tokens over %s grants; restore p50/p90 %s/%s ms; '
            'capture p50/p90 %s/%s ms; %s turns/h; RSS max %s GB; CI pods %s' % (
                name, s['served'], s['turns'], s['ttft_p50'], s['ttft_p90'], s['ttft_continuation_p50'],
                s['ttft_continuation_p90'], s['turn_p50'], s['turn_p90'], s['hit_rate'], s['hit_rate_continuation'],
                s['hit_turns'], s['continuation_turns'], s['trim_loss_tokens'], s['grants'], s['restore_ms_p50'],
                s['restore_ms_p90'], s['capture_ms_p50'], s['capture_ms_p90'], s.get('turns_per_hour'),
                s['rss_gb_max'], s['ci_pods']))


def compare_phases(prefix, baseline):
    """TTFT and throughput with reuse against without it, per phase present in both."""
    out = {}
    for name, mine in sorted(prefix.items()):
        theirs = baseline.get(name)
        if not theirs:
            continue
        out[name] = dict(
            ttft_p50=(mine.get('ttft_continuation_p50'), theirs.get('ttft_continuation_p50')),
            ttft_p90=(mine.get('ttft_continuation_p90'), theirs.get('ttft_continuation_p90')),
            turns_per_hour=(mine.get('turns_per_hour'), theirs.get('turns_per_hour')))
    return out
