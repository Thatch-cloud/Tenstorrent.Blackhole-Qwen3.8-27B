"""Paired trace timing with initial-state restore outside every measured replay."""

import statistics
import time


def checkpoint_prefixes(rows, policy):
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16):
        raise ValueError("Expected T1/2/4/8/16")
    if policy == "all":
        return tuple(range(rows + 1))
    if policy == "end":
        return (rows,)
    if policy == "none":
        return ()
    raise ValueError("Unknown checkpoint policy")


def paired_replays(restore, synchronize, replay, validate, clock=time.perf_counter):
    samples = []
    for block in range(3):
        for arm in ("control", "candidate", "candidate", "control"):
            durations = []
            for _ in range(10):
                restore()
                synchronize()
                started = clock()
                replay(arm)
                synchronize()
                durations.append((clock() - started) * 1000)
            validate(arm)
            samples.append(dict(block=block, arm=arm, replays=10, mean_ms=statistics.mean(durations)))
    ratios = []
    for block in range(3):
        means = {arm: statistics.mean(sample["mean_ms"] for sample in samples
                                     if sample["block"] == block and sample["arm"] == arm)
                 for arm in ("control", "candidate")}
        ratios.append(means["control"] / means["candidate"])
    return dict(samples=samples, paired_ratios=ratios, median_paired_ratio=statistics.median(ratios),
                exact=True, scope="One GDN layer with reported checkpoint policy; no full-model tok/s")
