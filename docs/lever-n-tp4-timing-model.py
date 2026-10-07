# Lever N TP4 timing model (e / cm). Chunk time fitted to measured TTFT deltas (T3/T4, v388/v389):
# ~30k prompts: ~8.6 s per user incl. build; ~119.5k: ~36 s per user incl. build.
A, B = 0.42, 0.0025e-3      # s per 2048-chunk = A + B * position_tokens
BUILD = 1.9                 # engine build at the final step, s (e; 1.7 m at TP2 warm, 2-3 s quoted)
ROUND = 0.245               # 8-live round, s (owner's figure; 7 live about the same: padded blocks)
EPS = 0.04                  # transition overhead per chunk step, s (e)
TAU = 4.4
CH = 2048

def chunks(P):
    F = P // CH * CH
    tail = P - F
    out = [(s, CH) for s in range(0, F, CH)]
    if tail:
        out.append((F, tail))
    return out

def t_chunk(start, n):
    # tail chunks run the masked bucket: charge proportionally, floor 40%
    full = A + B * (start + n)
    return full if n == CH else full * max(0.4, n / CH)

def prefill(P):
    return sum(t_chunk(s, n) for s, n in chunks(P))

def check():
    for P in (30000, 119500):
        print('fit P=%d prefill %.1f s + build %.1f = %.1f s' % (P, prefill(P), BUILD, prefill(P) + BUILD))

def steps(P):
    """Lever N steps: 2048 each, final step = last full chunk + tail (window coalescing)."""
    cs = chunks(P)
    if len(cs) >= 2 and cs[-1][1] != CH:
        last = cs[-2:]
        cs = cs[:-2] + [('final', last)]
    else:
        cs = cs[:-1] + [('final', [cs[-1]])]
    out = []
    for c in cs:
        if c[0] == 'final':
            out.append(sum(t_chunk(s, n) for s, n in c[1]))
        else:
            out.append(t_chunk(*c))
    return out

def levern(P, mode, value, decoders_rounds=None):
    """returns (ttft, worst_gap, decode_share, rounds_given). mode 'r' = static rounds/chunk, 'f' = prefill share."""
    st = steps(P)
    t = 0.0
    rounds = 0
    worst = 0.0
    left = decoders_rounds
    for i, c in enumerate(st):
        final = i == len(st) - 1
        dur = c + EPS + (BUILD if final else 0.0)
        t += dur
        worst = max(worst, dur)
        if final:
            break
        if left is not None and left <= 0:
            continue
        if mode == 'r':
            k = value
        else:
            k = max(1, round(dur * (1 - value) / value / ROUND))
        if left is not None:
            k = min(k, left)
            left -= k
        t += k * ROUND
        rounds += k
    return t, worst, rounds

if __name__ == '__main__':
    check()
    P = 253920
    base = prefill(P) + BUILD
    print('cold 253,920 OFF: prefill %.1f s, TTFT %.1f s, decoders frozen %.1f s' % (prefill(P), base, base))
    print('  steps %d, max step %.2f s (final incl. tail), chunk at 250k %.2f s' % (len(steps(P)), max(steps(P)), t_chunk(249856, 2048)))
    for mode, v in (('r', 1), ('r', 2), ('f', 0.67), ('f', 0.5), ('f', 0.33)):
        ttft, worst, rounds = levern(P, mode, v)
        busy = ttft
        print('  ON %s=%s long decoders: TTFT %.1f s (+%.0f%%), rounds to decoders %d -> mean %.1f tok/s/seat in window, worst gap %.1f s'
              % (mode, v, ttft, 100 * (ttft / base - 1), rounds, rounds * TAU / ttft, worst))
    for left_tokens in (176, 543):
        r = int(round(left_tokens / TAU))
        for mode, v in (('r', 1), ('f', 0.5), ('f', 0.33)):
            ttft, worst, rounds = levern(P, mode, v, decoders_rounds=r)
            print('  ON %s=%s agent decoders (%d tok left = %d rounds): arrival TTFT %.1f s (+%.0f%%) ; decoders done after ~%s'
                  % (mode, v, left_tokens, r, ttft, 100 * (ttft / base - 1), '-'))
    for P in (30000, 65000, 119500):
        base = prefill(P) + BUILD
        for mode, v in (('r', 1), ('f', 0.5)):
            ttft, worst, rounds = levern(P, mode, v)
            print('P=%d OFF %.1f s ; ON %s=%s %.1f s (+%.0f%%), worst gap %.2f s, decoder rate %.1f tok/s' % (P, base, mode, v, ttft, 100*(ttft/base-1), worst, rounds*TAU/ttft))

def decoders_done(P, mode, value, need):
    st = steps(P)
    t, given = 0.0, 0
    for i, c in enumerate(st):
        final = i == len(st) - 1
        dur = c + EPS + (BUILD if final else 0.0)
        t += dur
        if final:
            break
        k = value if mode == 'r' else max(1, round(dur * (1 - value) / value / ROUND))
        k = min(k, need - given)
        t += k * ROUND
        given += k
        if given >= need:
            return t
    return t + (need - given) * ROUND

if __name__ == '__main__':
    P = 253920
    for need_tok in (176, 543):
        need = int(round(need_tok / TAU))
        off = prefill(P) + BUILD + need * ROUND
        print('agent decoders %d tok (%d rounds): OFF finish at %.1f s' % (need_tok, need, off), end='')
        for mode, v in (('r', 1), ('f', 0.5), ('f', 0.33)):
            print(' | %s=%s finish at %.1f s' % (mode, v, decoders_done(P, mode, v, need)), end='')
        print()
