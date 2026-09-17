"""Hardware trace gate for the simulator-certified causal mask refresher."""

import importlib.util
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wide', action='store_true')
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('mask_replay_gate', '/experiment-optimisation/sim/attention-mask-replay.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.main(hardware=True, wide=options.wide)
