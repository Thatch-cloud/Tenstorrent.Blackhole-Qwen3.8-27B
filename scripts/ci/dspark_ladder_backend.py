"""Explicit backend admission for the isolated ladder correctness probe."""

from feature_projection import require_projection_environment


def require_backend(environment, *, hardware, device_present):
    if type(hardware) is not bool or type(device_present) is not bool:
        raise ValueError('Explicit backend and device presence required')
    require_projection_environment(environment, hardware)
    if hardware:
        if (not device_present or environment.get('QWEN_LADDER_BACKEND') != 'hardware'
                or environment.get('QWEN_LADDER_CONTEXT') != '65536'
                or environment.get('QWEN_PROJECTION_LINKS') != '4'
                or any(environment.get(name) for name in (
                    'QWEN_SIM_ONLY', 'QWEN_SIM_CASE', 'QWEN_SIM_SHARED_BDF',
                    'QWEN_SIM_BOUNDED_MEMORY', 'QWEN_LADDER_SCORE_SMOKE'))):
            raise ValueError('Dedicated allocated 64K hardware probe without simulator flags required')
        return 'hardware'
    if (device_present or environment.get('QWEN_SIM_SHARED_BDF') != '1'
            or environment.get('QWEN_SIM_BOUNDED_MEMORY') != '1'
            or environment.get('QWEN_LADDER_BACKEND', 'simulator') != 'simulator'
            or any(environment.get(name) == '1' for name in (
                'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))):
        raise ValueError('Bounded device-free simulator required')
    return 'simulator'
