"""Explicit canary-only paired request scopes; ordinary serving is unchanged."""

from contextlib import nullcontext
import json
import os


ARMS = ('control', 'grouped', 'control', 'grouped', 'grouped', 'control')


class GatherExperiment:
    def __init__(self, admission, scope):
        self.admission, self.scope = admission, scope
        self.ordinal = 0
        self.active = False

    def create(self, factory):
        if self.active or self.ordinal >= len(ARMS):
            raise ValueError('Exclusive bounded grouped-gather comparison required')
        ordinal = self.ordinal
        arm = ARMS[ordinal]
        context = self.scope(self.admission) if arm == 'grouped' else nullcontext(None)
        audit = context.__enter__()
        self.active = True
        try:
            request = factory()
        except BaseException:
            context.__exit__(None, None, None)
            self.active = False
            raise
        original_close = request.close
        released = False

        def close(request_id):
            nonlocal released
            if released:
                return original_close(request_id)
            original_close(request_id)
            context.__exit__(None, None, None)
            if audit is not None and (not audit['restored'] or not audit['kernels']):
                raise ValueError('Candidate scope must execute and restore before acceptance')
            released = True
            self.active = False
            self.ordinal += 1
            print(json.dumps(dict(stage='gather_comparison_request', ordinal=ordinal,
                arm=arm, warmup=ordinal < 2, audit=audit, performance_qualified=False)), flush=True)

        request.close = close
        return request


def from_environment(directory, runtime):
    flag = os.environ.get('QWEN_GDN_GROUPED_GATHER_ABBA', '0')
    if flag == '0':
        return None
    if flag != '1' or any(os.environ.get(name) != '1' for name in
            ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED', 'QWEN_FAST_PHASE_TIMING')):
        raise ValueError('Explicit allocated instrumented canary comparison required')
    from mlp_block_stream_runtime import require_hardware
    from gdn_grouped_gather_gate import qualify
    from gdn_grouped_gather_scope import scoped_gather

    require_hardware(os.environ)
    return GatherExperiment(qualify('/canary/grouped-gather-evidence', directory, runtime), scoped_gather)
