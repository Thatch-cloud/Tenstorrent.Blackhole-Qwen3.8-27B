"""Do two established requests DECODE in one step, or one per step?

This decides how big T6 is. The fast decode contract is single-resident:

    if (scheduled.scheduled_new_reqs or scheduled.finished_req_ids
            or list(cached.req_ids) != [request_id] ...):
        raise ValueError('Single resident decode request at the exact scheduled frontier required')

and FastWorkerHook closes over ONE bridge, so one hook binds one request.

  - If TTScheduler hands out cached=['A'] then cached=['B'] on alternate steps,
    T6 is a registry: one hook, a dict of request id to bridge, dispatch per step.
    Each device execution stays batch 1 and the verifier row budget stays 16.
  - If it hands out cached=['A','B'] in ONE step, then a step carries two requests'
    proposals at once, execute_decode must run a batch-2 verify, and dflash must
    support batch > 1 with 2 x 16 = 32 verifier rows. That is a device-side change,
    not a bookkeeping one.

Also reports whether a batched step keeps both requests' proposals in
scheduled_spec_decode_tokens, since that is what a batched verify would consume.

CPU only: no device, no weights.
"""

import sys
from tempfile import TemporaryDirectory

from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import Request

from probe_prefill_during_decode import build, output, show


def scenario(label, scheduler_type):
    parameters = SamplingParams(temperature=0, max_tokens=256)
    proposals = list(range(101, 116))
    with TemporaryDirectory() as directory:
        scheduler = build(directory, 2, scheduler_type)
        # A prefills and starts speculating, then B prefills. Staggered on purpose:
        # probe 35436384975 showed simultaneous arrivals batch into one prefill step,
        # which the fast path cannot serve, so this is the reachable route to two
        # established requests.
        scheduler.add_request(Request('A', [42] * 3000, parameters, None))
        first = scheduler.schedule()
        show('%s A prefill' % label, first)
        scheduler.update_from_output(first, output(['A'], 100))
        scheduler.update_draft_token_ids(DraftTokenIds(['A'], [proposals]))

        scheduler.add_request(Request('B', [42] * 3000, parameters, None))
        second = scheduler.schedule()
        show('%s B arrives' % label, second)
        # Both, not either: stock schedules new=['B'] WITH cached=['A'], and
        # update_from_output raises KeyError for any scheduled id the output omits.
        ids = ([r.req_id for r in second.scheduled_new_reqs]
               + list(getattr(second.scheduled_cached_reqs, 'req_ids', []) or []))
        scheduler.update_from_output(second, output(ids, 100))
        for name in ids:
            scheduler.update_draft_token_ids(DraftTokenIds([name], [proposals]))

        steps = []
        for index in range(3):
            step = scheduler.schedule()
            _, cached = show('%s both live, step %d' % (label, index), step)
            steps.append((cached, dict(step.scheduled_spec_decode_tokens)))
            if not cached:
                break
            scheduler.update_from_output(step, output(cached, 200 + index))
            for name in cached:
                scheduler.update_draft_token_ids(DraftTokenIds([name], [proposals]))
        return steps


def main():
    scenario('stock ', None)
    try:
        from vllm_tt_plugin.scheduler import TTScheduler
    except BaseException as error:
        print('TTScheduler unavailable: %s' % error)
        TTScheduler = None
    steps = scenario('TTSched', TTScheduler) if TTScheduler else None

    print()
    print('VERDICT')
    if steps is None:
        print('  TTScheduler could not be imported, so serving behaviour is UNKNOWN')
        return 0
    widest = max([len(cached) for cached, _ in steps] or [0])
    for index, (cached, spec) in enumerate(steps):
        print('  step %d cached=%s spec_requests=%s' % (index, cached, sorted(spec)))
    if widest > 1:
        print('  TTScheduler BATCHES two decodes into one step, so the fast decode path')
        print('  must run a batch-2 verify and dflash needs batch > 1 with 2 x 16 rows.')
        print('  T6 is a device-side change, not a registry of hooks.')
    elif widest == 1:
        print('  TTScheduler issues ONE decode request per step, so each device')
        print('  execution stays batch 1 at 16 verifier rows. T6 is then bookkeeping:')
        print('  one hook holding request id to bridge, dispatching on the step id.')
    else:
        print('  no decode step was produced, so this probe measured nothing')
    return 0


if __name__ == '__main__':
    sys.exit(main())
