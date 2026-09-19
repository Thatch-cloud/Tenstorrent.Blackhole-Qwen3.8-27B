"""Where can the one-in-flight prefill rule actually be installed?

Probe 35436384975 measured TTScheduler batching SIMULTANEOUS prefills into one
step, `new=['A','B']`. The fast prefill path takes one capture and executes one
prompt, so that step cannot be served and the scheduler has to hand out at most
one fresh prompt per step.

Where that rule goes is not guessable from outside the image. serving_plugin_patch
edits plugin source textually against a pinned revision, and a patch written
against source I have not read is how earlier mistakes happened. So this reports
the facts a correct patch needs:

  - is TTScheduler a subclass of vLLM's Scheduler, and what does it override?
  - does the platform expose a scheduler class hook (get_scheduler_cls or similar)?
  - does SchedulerConfig carry a scheduler_cls that a config could simply point at?
  - what is the source of the method that picks up waiting requests?

If scheduler_cls is settable, the rule is a subclass and no source patch is needed.

CPU only: no device, no weights.
"""

import inspect
import sys


def show(label, value):
    print('%-44s %s' % (label, value))


def main():
    try:
        from vllm_tt_plugin.scheduler import TTScheduler
    except BaseException as error:
        print('VERDICT')
        print('  TTScheduler could not be imported: %s' % error)
        return 0
    from vllm.v1.core.sched.scheduler import Scheduler

    show('TTScheduler bases', [base.__name__ for base in TTScheduler.__bases__])
    show('subclasses vLLM Scheduler', issubclass(TTScheduler, Scheduler))
    overrides = sorted(name for name, value in vars(TTScheduler).items()
                       if callable(value) and not name.startswith('__'))
    show('TTScheduler overrides', overrides)

    for name in ('schedule', '_schedule_prefill_only', '_schedule_decode_only'):
        method = getattr(TTScheduler, name, None)
        show('has %s' % name, 'own' if name in vars(TTScheduler) else bool(method))

    waiting_users = []
    for name in overrides:
        try:
            source = inspect.getsource(getattr(TTScheduler, name))
        except BaseException:
            continue
        if 'waiting' in source:
            waiting_users.append((name, source.count('waiting'), len(source.splitlines())))
    show('overrides mentioning waiting', waiting_users)

    from vllm.config import SchedulerConfig

    fields = getattr(SchedulerConfig, '__dataclass_fields__', {})
    show('SchedulerConfig has scheduler_cls', 'scheduler_cls' in fields)
    if 'scheduler_cls' in fields:
        show('scheduler_cls default', fields['scheduler_cls'].default)

    platform = None
    try:
        from vllm_tt_plugin.platform import TTPlatform as platform
    except BaseException as error:
        show('platform import', 'failed: %s' % error)
    if platform is not None:
        hooks = sorted(name for name in dir(platform) if 'sched' in name.lower())
        show('platform scheduler hooks', hooks)
        for name in hooks:
            try:
                show('  %s source lines' % name,
                     len(inspect.getsource(getattr(platform, name)).splitlines()))
            except BaseException:
                pass

    # The rule will hide extra waiting requests before delegating, so the exact
    # shape of the method and of the waiting container decide whether that is safe.
    for name in ('_schedule_prefill_only', '_has_pending_prefill'):
        try:
            print('----- %s -----' % name)
            print(inspect.getsource(getattr(TTScheduler, name)))
        except BaseException as error:
            show('source of %s' % name, 'unavailable: %s' % error)
    try:
        from vllm.v1.core.sched.request_queue import create_request_queue
        show('request queue factory', create_request_queue)
    except BaseException as error:
        show('request queue factory', 'unavailable: %s' % error)
    show('Scheduler.waiting annotation', getattr(Scheduler, '__annotations__', {}).get('waiting'))

    print()
    print('VERDICT')
    if 'scheduler_cls' in fields:
        print('  SchedulerConfig carries scheduler_cls, so the one-in-flight rule can be a')
        print('  SUBCLASS of TTScheduler pointed at by config - no source patch, and the')
        print('  pinned plugin revision stays untouched.')
    elif waiting_users:
        print('  No scheduler_cls, but %s mention waiting, so the rule belongs there'
              % ', '.join(name for name, _, _ in waiting_users))
        print('  and serving_plugin_patch needs a textual patch against that method.')
    else:
        print('  Neither a scheduler_cls nor an override touching waiting was found, so')
        print('  the rule has no obvious seam and the approach needs rethinking.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
