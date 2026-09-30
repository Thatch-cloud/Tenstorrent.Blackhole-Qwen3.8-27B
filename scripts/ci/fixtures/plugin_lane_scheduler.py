class LaneScheduler(object):
    def _local_prefill_intent(self, sched: TTScheduler) -> int:
        """Whether this lane *wants* to prefill this step (1) or not (0).

        A lane wants to prefill when it has queued requests and either nothing
        running (so it must prefill to make progress) or spare capacity to admit
        more alongside its running decodes.

        A partial prefill continuation always wants to prefill, even with the
        lane at capacity: it already holds one of those slots and only a
        prefill step can advance it.

        ``skipped_waiting`` holds prefill requests blocked on grammar
        compilation. We want to try to schedule them, because otherwise the
        base scheduler won't revisit and promote them - decode intent hides
        that queue.
        """
        has_waiting = bool(sched.waiting) or bool(sched.skipped_waiting)
        has_running = bool(sched.running)
        has_partial_prefill = any(r.is_prefill_chunk for r in sched.running)
        has_capacity = len(sched.running) < self._per_lane_max
        return int(
            has_partial_prefill or (has_waiting and ((not has_running) or has_capacity))
        )

    def _negotiate_forced_mode(self) -> TTSchedulingMode:
        """Pick the single mode (prefill- or decode-only) all lanes will run.

        The device executes all lanes together and cannot mix prefill with
        decode, so the lanes must agree. If *any* lane wants to prefill, the
        whole step is prefill-only; otherwise it is decode-only. Lanes without
        work for the chosen mode simply contribute an empty batch.
        """
        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)
