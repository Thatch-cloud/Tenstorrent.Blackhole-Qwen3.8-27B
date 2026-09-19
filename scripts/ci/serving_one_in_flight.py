"""One fresh prompt per prefill step, as a scheduler subclass.

Probe 35436384975 measured TTScheduler batching SIMULTANEOUS prefills into one
step, `new=['A','B']`. The fast prefill path takes one capture and executes one
prompt, so that step cannot be served at all - and run 35436193682 failed on
exactly it, at `len(scheduled_new_reqs) != 1`.

Probe 35440389384 then measured `SchedulerConfig.scheduler_cls` existing and
defaulting to None, so this is a SUBCLASS rather than a textual patch of the
pinned plugin.

How the cap works, from the plugin's own source (probe 35440459594):

    def _schedule_prefill_only(self) -> SchedulerOutput:
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]
        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        ...

The base waiting loop admits until `len(self.running)` reaches
`max_num_running_reqs`, and TTScheduler has already set `running` to the partial
prefills. So capping the effective value at `len(partial_prefills) + 1` leaves
room for exactly one fresh prompt.

It is set BEFORE delegating and compensated for the subtraction the plugin is
about to do, rather than by copying the method body. Copying it would silently
diverge the moment the pinned plugin changed, and the whole point of using
scheduler_cls is to leave that source alone.
"""


def one_in_flight_scheduler(base=None):
    """Build the subclass against whichever scheduler the plugin provides."""
    if base is None:
        from vllm_tt_plugin.scheduler import TTScheduler as base

    class OneInFlightScheduler(base):
        """Admits at most one fresh prompt per prefill step."""

        def _schedule_prefill_only(self):
            saved = self.max_num_running_reqs
            decodes = sum(1 for request in self.running if not request.is_prefill_chunk)
            partials = len(self.running) - decodes
            try:
                # The plugin will subtract `decodes` from whatever it reads here, so
                # add it back: the value it computes becomes partials + 1.
                self.max_num_running_reqs = min(saved, partials + 1 + decodes)
                return super()._schedule_prefill_only()
            finally:
                self.max_num_running_reqs = saved

    return OneInFlightScheduler


def effective_capacity(max_num_running_reqs, decodes, partials):
    """What the plugin computes for the waiting loop, once capped.

    Kept separate so the arithmetic can be checked without a scheduler: the plugin
    computes max(0, self.max_num_running_reqs - decodes) after this class has
    already written a capped value into that attribute.
    """
    if any(type(value) is not int or value < 0
           for value in (max_num_running_reqs, decodes, partials)):
        raise ValueError('Non-negative integer scheduler capacities required')
    capped = min(max_num_running_reqs, partials + 1 + decodes)
    return max(0, capped - decodes)


def install(vllm_config, scheduler=None):
    """Point the fast path's config at the one-in-flight scheduler.

    Only when the fast path is requested: this is part of that path's contract,
    not a change to how the plugin schedules by default.
    """
    scheduler_config = vllm_config.scheduler_config
    if getattr(scheduler_config, 'scheduler_cls', None) not in (None, ''):
        raise ValueError('A scheduler class is already selected for this config')
    scheduler_config.scheduler_cls = scheduler or one_in_flight_scheduler()
    return scheduler_config.scheduler_cls
