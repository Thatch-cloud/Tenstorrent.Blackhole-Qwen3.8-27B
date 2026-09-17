"""Restore the simulated slice host operation only inside the disposable DSpark integration container."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from dspark_hardware_gate import digest


REVISION = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
SOURCE = 'ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp'
IMAGE_SHA256 = '817b571dc619eef7af7988ad90e3eda4a89632af3477ca935048b05dc52aea6f'
SIMULATED_SHA256 = '06208fab2ea3dbdc6a49f022af94c4ee7fe0b30e4d6125197cd318d81bfe6da3'


def restore(root):
    root = Path(root)
    path = root/SOURCE
    before = digest(path)
    if before not in (IMAGE_SHA256,SIMULATED_SHA256):
        raise ValueError('Unknown slice implementation; refusing source replacement')
    if before!=SIMULATED_SHA256:
        result = subprocess.run(['git','-C',str(root),'show',REVISION+':'+SOURCE],check=True,stdout=subprocess.PIPE)
        if hashlib.sha256(result.stdout).hexdigest()!=SIMULATED_SHA256:
            raise ValueError('Git source does not match the exact simulated slice implementation')
        path.write_bytes(result.stdout)
    after = digest(path)
    if after!=SIMULATED_SHA256:
        raise ValueError('Restored slice source changed')
    return dict(source=SOURCE,before=before,after=after,revision=REVISION,
        binary_rebuild_or_verified_restore_required=True,
        scope='Disposable DSpark FC/32-row backbone integration only; not serving or target runtime promotion')


if __name__=='__main__':
    if (os.environ.get('QWEN_HARDWARE_TESTS')!='1' or os.environ.get('QWEN_CARDS_ALLOCATED')!='1'
            or os.environ.get('TT_METAL_HOME')!='/opt/tt-metal' or os.environ.get('TT_METAL_SIMULATOR')
            or Path(__file__).parent!=Path('/experiment-scripts/ci')):
        raise ValueError('Only the allocated disposable hardware container may restore native sources')
    Path('/experiment/results/dspark-native-restore.json').write_text(json.dumps(restore('/opt/tt-metal'),indent=2)+'\n')
