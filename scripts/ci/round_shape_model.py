"""round_shape_model: what an ADAPTIVE speculation shape at 6 to 8 live users would buy on the s5 stack. An ESTIMATE, built from measured rounds; nothing here has run on a card.

    python3 scripts/ci/round_shape_model.py            the tables (stdlib only, Python 3.7 syntax)

THE MODEL. A round is  wall = entry + gap  +  verify trace(block, context)  +  step extras (verify glue, commit enqueue)  +  early draft (shape, live).  A user commits
E[min(e, rows)] tokens a round, e being the length of the accepted prefix plus the bonus token, and its rate is that over the wall; the aggregate is the sum over the live users, the
slowest user is the lowest rate. The acceptance of a user is the geometric curve P(e >= k) = rho^(k - 1), rho fitted to the user's MEASURED mean under the octo block's 8-row cap
(E[min(e, 8)] = (1 - rho^8) / (1 - rho)); the pooled measured curves (round_probe_report, ACCEPTANCE) are geometric to within about 3% of E[min(e, 8)] at 8 rows and the 16-row curves
within 3% as well, with a slightly heavier tail than the fit (so the fit is a little pessimistic for any shape that adds rows).

LABELS. (M) measured: a median of the s5 C1 and L1 runs (round_probe_report on their container logs; the two runs agree within 2%); (E) estimate, with its basis; (G) a GAP, a number
no log holds, taken at the value stated and varied in the sensitivity.

Measured inputs (ms):
  verify trace, M3 block (4 segments of 16 rows):  42.4 at 3.6k (M, 1 to 4 live: the trace does not care how many segments are live), 48.0 at ~70k, 58.9 at 110k and above (M).
  verify trace, octo block (8 segments of 8 rows): 52.2 at 3.6k with 8 live (M); 76.3 at ~130k with 6 live (M: 78.3 and 74.2). 8 live at depth: taken equal (E: the attention of the
  bundle that holds the idle segments is set by its longest live user, as the M3 block's is: one user at 210k costs the same 58.9 as three at 130k).
  step extras (verify glue + commit enqueue + the rest of the step): M3 2.7, octo 6.0 (M).     entry + gap: M3 1.3, octo 1.5 (M).
  early draft: M3 block 13.6 (1 to 2 live), 16.5 (4 live); octo pass at 8 live 21.2 to 22.9; the two quads cut to 7 rows at 6 to 7 live: 39.0 to 41.3 at 3.6k, 30.5 to 30.7 at ~130k (M);
  two M3 blocks (3 to 5 live spread over both): 22 to 37 (M).  The draft does NOT grow with the target's context in the quad path; the pair path grows ~10 ms from 4k to 100k (M, 2 live);
  whether the octo pass grows is a GAP, varied below by +0 and +10 ms.
  acceptance: mean tokens a user commits under the 8-row cap, the 8 prompts of concurrent8_steady (M): 3.39 3.36 3.65 3.24 5.19 3.22 2.92 2.87.
Estimates:
  per removed idle segment of a pass: -2.45 ms (E: 42.4 for 4 segments against 52.2 for 8 gives 32.6 fixed + 2.45 a segment; the rows differ, 16 against 8, and rows are nearly free).
  the octo draft pass costs 2% of the tokens against the quads cut to 7 rows (E: tp4-octo3.md section 4, measured on the Lever N stack; the s5 logs cannot say, tau moves with the output
  position by more than that), varied to 10% in the sensitivity.
  verify trace of the octo block at ~70k: 61.0 (E: the M3 block's rise over its 3.6k value, 5.6 ms, scaled by the octo block's rise at 130k over the M3 block's, 26.1 / 16.5).

THE SHAPES (all exact: the target's argmax is the only token ever committed; only tau and time move):
  K0  today: the octo block at 6 or more live (8 rows a segment); the octo draft pass at 8 live, the two quads cut to 7 rows at 6 and 7.
  K1  the octo draft pass at 6 and 7 live as well (idle seats masked in the pass: code that today refuses with fewer than eight seats live; the draft only changes tau).
  K2  K1 plus a block of exactly n segments of floor(64 / n) rows (6 x 10, 7 x 9): a new trace per n, about 0.5 GB of engine memory each on a stack whose octo plan has a 5 MB margin.
  K3  K1 plus rows reallocated among the 8 segments by the users' own curves (a trace per allocation: only a few presets are possible; this is the bound).
  K4  K1 with the lowest-acceptance users NOT speculating (1 row) and their rows given to the rest (m = 1, 2).
  K5  two M3 blocks (4 + 4 users, 16 rows): the shape the octo block replaced.
  At five live (the octo block refuses it today: a block holds two idle segments, five live leaves three, and the NP1 scratch target that served a lone user is not wired to the octo block):
  K0 is the two-M3-block round as measured (4 + 1 users, 16 rows a user) and K1 the octo block with its draft pass and three idle segments (8 rows a user).
"""
import itertools
import statistics

SEGMENT_MS = 2.45
OCTO_PASS_TAU = 0.98
OCTO_PASS_TAU_LOW = 0.90

CONTEXTS = ('3.6k', '64k', '128k')
VERIFY = {('octo', '3.6k'): 52.2, ('octo', '64k'): 61.0, ('octo', '128k'): 76.3,
          ('m3', '3.6k'): 42.4, ('m3', '64k'): 48.0, ('m3', '128k'): 58.9}
EXTRAS = {'octo': 6.0, 'm3': 2.7}
ENTRY_GAP = {'octo': 1.5, 'm3': 1.3}
DRAFT_OCTO_PASS = 22.0
DRAFT_QUADS_CUT = {'3.6k': 40.0, '64k': 35.0, '128k': 30.6}
DRAFT_TWO_QUADS = 30.0
# live users spread over BOTH M3 blocks (4 + 1 at five live), the whole round as measured: 165.9 at ~130k (M, 286 rounds); at 3.6k the rounds of three to four users in two blocks (M: step 91.2,
# draft 27.1, entry + gap 1.3).
TWO_BLOCK_WALL = {'3.6k': 119.6, '64k': 142.0, '128k': 165.9}
TAU8 = (3.39, 3.36, 3.65, 3.24, 5.19, 3.22, 2.92, 2.87)
POOLED_TAU_DEEP = 3.53


def expected_prefix(rho, rows):
    """E[min(e, rows)] for P(e >= k) = rho ** (k - 1)."""
    if rows <= 0:
        return 0.0
    if abs(1.0 - rho) < 1e-12:
        return float(rows)
    return (1.0 - rho ** rows) / (1.0 - rho)


def fit_rho(mean_at_cap, cap=8):
    """The rho whose E[min(e, cap)] is `mean_at_cap` (bisection; the mean must lie in [1, cap])."""
    low, high = 0.0, 1.0 - 1e-9
    mean_at_cap = min(max(mean_at_cap, 1.0), float(cap))
    for _ in range(80):
        middle = (low + high) / 2.0
        if expected_prefix(middle, cap) < mean_at_cap:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def user_curves(context):
    """[rho of each of the 8 measured users] at a context; at depth the spread is kept and the mean moves to the pooled agent-mix value."""
    shift = 1.0 if context == '3.6k' else POOLED_TAU_DEEP / statistics.mean(TAU8)
    return [fit_rho(mean * shift) for mean in TAU8]


def subsets(count):
    """Every choice of `count` users of the 8, as index tuples."""
    return list(itertools.combinations(range(8), count))


def allocate_rows(rhos, total, floor=1, cap=16):
    """Rows a user (integer, floor to cap, summing to `total`) maximising the sum of E[min(e, rows)]: the greedy on the marginal gain rho ** rows is exact for concave gains."""
    rows = [floor] * len(rhos)
    for _ in range(total - floor * len(rhos)):
        best = max((index for index in range(len(rhos)) if rows[index] < cap), key=lambda index: rhos[index] ** rows[index])
        rows[best] += 1
    return rows


def draft_ms(shape, live, context, octo_growth):
    if shape in ('octo-pass',):
        return DRAFT_OCTO_PASS + octo_growth * (0.0 if context == '3.6k' else 1.0)
    if shape == 'quads-cut':
        return DRAFT_QUADS_CUT[context]
    return DRAFT_TWO_QUADS


def evaluate(name, live, context, users, rows, verify, extras, entry_gap, draft, tau_scale=1.0):
    """-> dict(name, live, context, wall, aggregate, slowest, user): one configuration, averaged over every choice of `live` users of the 8."""
    walls = verify + extras + entry_gap + draft
    aggregates, slowests, per_user = [], [], []
    for chosen in subsets(live):
        rhos = [users[index] for index in chosen]
        row_list = rows(rhos) if callable(rows) else [rows] * live
        tokens = [expected_prefix(rho, r) * tau_scale for rho, r in zip(rhos, row_list)]
        aggregates.append(sum(tokens) / walls * 1000.0)
        slowests.append(min(tokens) / walls * 1000.0)
        per_user.append(statistics.mean(tokens) / walls * 1000.0)
    return dict(name=name, live=live, context=context, wall=walls, aggregate=statistics.mean(aggregates), slowest=statistics.mean(slowests), user=statistics.mean(per_user))


def shapes(live, context, octo_growth=0.0, octo_pass_tau=OCTO_PASS_TAU):
    """[evaluation] of K0 to K5 at `live` users and a context."""
    users = user_curves(context)
    verify_octo, verify_m3 = VERIFY[('octo', context)], VERIFY[('m3', context)]
    out = []
    quads_gain = 1.0 / octo_pass_tau
    if live == 5:
        out.append(evaluate('K0 today (two M3 blocks, 4+1)', live, context, users, 16, TWO_BLOCK_WALL[context], 0.0, 0.0, 0.0))
        out.append(evaluate('K1 octo block + pass (3 idle seg)', live, context, users, 8, verify_octo, EXTRAS['octo'], ENTRY_GAP['octo'], draft_ms('octo-pass', live, context, octo_growth)))
        return out
    if live == 8:
        out.append(evaluate('K0 today (octo pass)', live, context, users, 8, verify_octo, EXTRAS['octo'], ENTRY_GAP['octo'], draft_ms('octo-pass', live, context, octo_growth)))
    else:
        out.append(evaluate('K0 today (quads cut to 7)', live, context, users, 8, verify_octo, EXTRAS['octo'], ENTRY_GAP['octo'], draft_ms('quads-cut', live, context, 0), quads_gain))
        out.append(evaluate('K1 octo pass at %d live' % live, live, context, users, 8, verify_octo, EXTRAS['octo'], ENTRY_GAP['octo'], draft_ms('octo-pass', live, context, octo_growth)))
        segments = live
        out.append(evaluate('K2 K1 + %d x %d-row block' % (segments, 64 // segments), live, context, users, 64 // segments,
                            verify_octo - SEGMENT_MS * (8 - segments), EXTRAS['octo'], ENTRY_GAP['octo'], draft_ms('octo-pass', live, context, octo_growth)))
    if live == 8:
        out.append(evaluate('K3 K0 + rows by curve (bound)', live, context, users, lambda rhos: allocate_rows(rhos, 64, floor=4, cap=16), verify_octo, EXTRAS['octo'], ENTRY_GAP['octo'],
                            draft_ms('octo-pass', live, context, octo_growth)))
        for skipped in (1, 2):
            def rows_skip(rhos, skipped=skipped):
                order = sorted(range(len(rhos)), key=lambda index: rhos[index])
                rest = [index for index in range(len(rhos)) if index not in order[:skipped]]
                given = allocate_rows([rhos[index] for index in rest], 64 - skipped, floor=1, cap=16)
                result = [1] * len(rhos)
                for index, value in zip(rest, given):
                    result[index] = value
                return result

            out.append(evaluate('K4 %d lowest not speculating' % skipped, live, context, users, rows_skip, verify_octo, EXTRAS['octo'], ENTRY_GAP['octo'], draft_ms('octo-pass', live, context, octo_growth)))
    two = 2.0 * (verify_m3 + EXTRAS['m3'])
    out.append(evaluate('K5 two M3 blocks (16 rows)', live, context, users, 16, two, 0.0, ENTRY_GAP['m3'], draft_ms('two-quads', live, context, 0)))
    return out


def table(octo_growth=0.0, octo_pass_tau=OCTO_PASS_TAU):
    lines = []
    for context in CONTEXTS:
        for live in (8, 7, 6, 5):
            results = shapes(live, context, octo_growth, octo_pass_tau)
            base = results[0]
            for result in results:
                lines.append('%5s live %d  %-34s wall %6.1f ms  aggregate %5.0f (x%.2f)  slowest user %5.1f (x%.2f)  a user %5.1f' % (
                    context, live, result['name'], result['wall'], result['aggregate'], result['aggregate'] / base['aggregate'], result['slowest'],
                    result['slowest'] / base['slowest'], result['user']))
            lines.append('')
    return '\n'.join(lines)


def main():
    print('ESTIMATE (see the module docstring for the measured inputs and every assumption). Ratios are against K0 at the same live count and context.')
    print('\nCENTRAL: octo pass tau %.2f of the quads, octo-pass draft flat with context' % OCTO_PASS_TAU)
    print(table())
    print('SENSITIVITY: the octo pass loses 10% of the tokens, and its draft grows 10 ms at depth')
    print(table(octo_growth=10.0, octo_pass_tau=OCTO_PASS_TAU_LOW))


if __name__ == '__main__':
    main()
