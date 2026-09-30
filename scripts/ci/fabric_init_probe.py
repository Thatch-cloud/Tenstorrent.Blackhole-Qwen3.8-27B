"""Does fabric firmware initialise across the two cards?

vLLM's engine died at FabricFirmwareInitializer::wait_for_fabric_router_sync with a
remote ethernet handshake timeout. This isolates that: open a 1x2 mesh with fabric
ENABLED and report whether the routers sync. Prefetch work never caught this because
the capability probe sets FabricConfig.DISABLED and the matmul arms never needed
fabric firmware.

Exit 0 means fabric came up, 1 means it did not. Opens and closes the cards once.
"""

import argparse
import json
import os
import sys
import traceback

BEGIN = '<<<FABRIC_PROBE_JSON_BEGIN>>>'
END = '<<<FABRIC_PROBE_JSON_END>>>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None,
                        help='FabricConfig name; default tries the 1D variants in turn')
    options = parser.parse_args()
    report = dict(scope=__doc__, fabric_up=False, attempts=[])
    try:
        import ttnn
        available = [name for name in dir(ttnn.FabricConfig) if not name.startswith('_')]
        report['fabric_config_values'] = available
        # DISABLED would trivially pass and prove nothing.
        candidates = ([options.config] if options.config else
                      [n for n in ('FABRIC_1D', 'FABRIC_1D_RING', 'FABRIC_2D')
                       if n in available])
        report['candidates'] = candidates
        for name in candidates:
            attempt = dict(config=name)
            mesh = None
            try:
                ttnn.set_fabric_config(getattr(ttnn.FabricConfig, name))
                mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
                attempt['opened'] = True
                attempt['devices'] = mesh.get_num_devices()
                report['fabric_up'] = True
            except BaseException as error:
                attempt['error'] = str(error)[:900]
                attempt['kind'] = type(error).__name__
            finally:
                if mesh is not None:
                    try:
                        ttnn.close_mesh_device(mesh)
                        attempt['closed'] = True
                    except BaseException as error:
                        attempt['close_error'] = str(error)[:300]
                try:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                except BaseException:
                    pass
            report['attempts'].append(attempt)
            if report['fabric_up']:
                break
    except BaseException:
        report['fatal'] = traceback.format_exc(limit=6)[-1200:]
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    sys.stdout.flush()
    os._exit(0 if report.get('fabric_up') else 1)


if __name__ == '__main__':
    main()
