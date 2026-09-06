"""Hardware trace gate for the simulator-certified causal mask refresher."""

import importlib.util


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('mask_replay_gate', '/experiment-optimisation/sim/attention-mask-replay.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.main(hardware=True)
