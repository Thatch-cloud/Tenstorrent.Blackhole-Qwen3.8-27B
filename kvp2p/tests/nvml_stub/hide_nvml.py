"""Test-only: make vLLM's platform probe find no NVIDIA driver.

vLLM auto-detects the CUDA platform when NVML reports a GPU and then needs its compiled extensions. These tests run the
real scheduler classes with no device, as the CPU CI job does on a runner with no GPU; on a host that has a GPU driver the
probe has to be told there is none. ``sitecustomize.py`` next to this file calls ``install`` so that the subprocess vLLM
starts to inspect a model architecture (it inherits PYTHONPATH) sees the same thing. Set TTKVP2P_TEST_HIDE_NVML=0 to leave
the probe alone.
"""

import os
import sys
import types


def install() -> None:
    if os.environ.get('TTKVP2P_TEST_HIDE_NVML', '1') == '0' or 'vllm.third_party.pynvml' in sys.modules:
        return
    stub = types.ModuleType('vllm.third_party.pynvml')

    class NVMLError(Exception):
        pass

    def nvmlInit():
        raise NVMLError('hidden by the ttkvp2p tests')

    stub.NVMLError = NVMLError
    stub.nvmlInit = nvmlInit
    sys.modules['vllm.third_party.pynvml'] = stub
