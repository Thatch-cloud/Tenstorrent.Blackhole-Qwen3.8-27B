from pathlib import Path
import subprocess
import tempfile
import unittest


class SimulatorAssetCacheTests(unittest.TestCase):
    def test_verified_hit_and_corrupt_download_rejection(self):
        source = Path(__file__).with_name('run-simulator.sh').read_text()
        function = source.split('fetch_simulator_asset() {', 1)[1].split('\nfetch_simulator_asset https:', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            script = 'set -euo pipefail\ncache="$1"\nfetch_simulator_asset() {' + function + '''
digest=$(printf verified | sha256sum | cut -d ' ' -f 1)
curl() { printf verified > "${@: -1}"; }
fetch_simulator_asset unused "$digest" "$cache/first"
curl() { return 99; }
fetch_simulator_asset unused "$digest" "$cache/second"
cmp "$cache/first" "$cache/second"
printf corrupted > "$cache/simulator-assets/$digest"
curl() { printf wrong > "${@: -1}"; }
if fetch_simulator_asset unused "$digest" "$cache/third"; then exit 1; fi
test ! -e "$cache/third"
'''
            subprocess.run(['bash', '-c', script, 'asset-test', directory], check=True,
                capture_output=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
