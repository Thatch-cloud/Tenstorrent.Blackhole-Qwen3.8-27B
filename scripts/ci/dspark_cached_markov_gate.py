"""Pinned simulator admission for cached Markov hardware experiments, not speed."""

import hashlib
import json
from pathlib import Path


RUN = 34735206013
REPORTS = {
    'dspark-cached-markov': '2084242137fd239a0a46e55810414cbb5cc8e411c2e01e212d025f6f295ac620',
    'markov-cache-pipeline': '4293428e4330eaeeced405ebbf295c9e565664f6a7112f943c0c5025ad6c87d1',
    'markov-cache-pipeline-4992': 'ec5f4e4b528f0f7b877aa8419d848b29df85007a6b776e10614feef25aaa8760',
    'markov-cache-pipeline-3712': 'ac8422725838000762d11f1822d61ee5525476176f1bc73819e60f58e9ed20ff',
}


def qualify(evidence_root, source_root):
    evidence_root, source_root = Path(evidence_root), Path(source_root)
    sources = {}
    for name, expected in REPORTS.items():
        payload = (evidence_root / (name + '.json')).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Exact accepted cache simulator report required: ' + name)
        report = json.loads(payload)
        if report.get('passed') is not True or report.get('closed_cleanly') is not True:
            raise ValueError('Successful clean cache simulator required')
        if report['sources'] != report['sources_after']:
            raise ValueError('Simulator sources changed during execution')
        for filename, checksum in {**report['sources'], **report['build']['builders']}.items():
            if Path(filename).name != filename:
                raise ValueError('Only local source basenames are admitted')
            actual = hashlib.sha256((source_root / filename).read_bytes()).hexdigest()
            if actual != checksum or sources.get(filename, checksum) != checksum:
                raise ValueError('Cache simulator source changed: ' + filename)
            sources[filename] = checksum
    return dict(simulator_run=RUN, reports=dict(REPORTS), sources=sources,
        full_vocabulary_qualified=False, performance_qualified=False,
        hardware_correctness_required=True)
