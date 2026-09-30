"""Validate every T32 component against the restored runtime before loading weights."""

import json
import os
from pathlib import Path

from dflash_t32_native_scope import admit
from gdn_shared_qk_t32_gate import qualify as qualify_gdn
from gdn_direct_window_t32_gate import qualify as qualify_windows
from mlp_down_grid_t32_gate import qualify as qualify_down
from mlp_block_stream_t32_gate import qualify as qualify_stream
from mlp_register_epilogue_gate import qualify as qualify_register


def evidence_paths(directory):
    return {name: Path(directory) / 't32-evidence' / name
        for name in ('attention', 'cache', 'gdn', 'windows', 'down', 'stream')}


def preload(directory, runtime):
    directory = Path(directory)
    validate_fusion_routes(directory)
    evidence = evidence_paths(directory)
    register = qualify_register(directory, directory / 'register-epilogue-evidence', runtime_root=runtime)
    admissions = dict(proposals=admit(evidence['attention'], evidence['cache'], directory, runtime),
        gdn=qualify_gdn(evidence['gdn'], directory, runtime),
        windows=qualify_windows(evidence['windows'], directory, runtime),
        down=qualify_down(evidence['down'], directory, runtime),
        stream=qualify_stream(directory, evidence['stream'], register))
    return dict(passed=True, stage='preload', weights_loaded=False, device_execution=False,
        hardware_qualified=False, performance_qualified=False, admissions=admissions)


def validate_fusion_routes(directory):
    import fused_t16_scope

    control = getattr(fused_t16_scope, 'FusedT16Arm', None)
    candidate = getattr(fused_t16_scope, 'FusedT32Arm', None)
    if (Path(fused_t16_scope.__file__).resolve() != (Path(directory) / 'fused_t16_scope.py').resolve()
            or not isinstance(control, type) or not isinstance(candidate, type)
            or not issubclass(candidate, control) or control.token_rows != 16 or candidate.token_rows != 32):
        raise ValueError('Staged width-aware T16/T32 fusion routes required before loading weights')


if __name__ == '__main__':
    print(json.dumps(preload(Path(__file__).parent, os.environ['TT_METAL_HOME']), indent=2))
