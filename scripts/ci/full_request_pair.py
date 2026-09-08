"""Matched actual requests; each arm owns a fresh prefill and closed trace lifetime."""

import math


def summarize_requests(requests, *, arm_key='norm_batch'):
    if arm_key not in ('norm_batch', 'attention_replay', 'attention_wide', 'lookup_cap', 'sampling_links'):
        raise ValueError('Known matched request experiment required')
    if len(requests) != 4 or [entry[arm_key] for entry in requests] != [False, True, True, False]:
        raise ValueError('One complete control/candidate/candidate/control request block required')
    reference = requests[0]
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size',
                'committed_decode_tokens', 'proposed', 'accepted')
    if arm_key == 'lookup_cap':
        identity = identity[:-2]
    routing = ('rows', 'source', 'accepted', 'committed', 'match_length', 'position', 'input_tokens')
    expected_blocks = [tuple(block[key] for key in routing) for block in reference['blocks']]
    for entry in requests:
        if type(entry[arm_key]) is not bool:
            raise ValueError('Explicit boolean arm selection required')
        if arm_key == 'sampling_links':
            if (entry.get('norm_batch') is not True or any(entry.get(key) is not False for key in
                    ('family_routing', 'attention_replay', 'attention_mask_once'))
                    or entry.get('replay_group_rows') != 4
                    or type(entry.get('lookup_max_rows')) is not int or entry['lookup_max_rows'] != 8
                    or type(entry.get('sampler_num_links')) is not int
                    or entry['sampler_num_links'] != (4 if entry[arm_key] else 1)
                    or not entry.get('fabric_sources') or entry['fabric_sources'] != reference.get('fabric_sources')
                    or entry.get('ended_with_eos') is not True):
                raise ValueError('Sampling comparison requires complete matched T8 coding requests with audited links')
        if arm_key == 'lookup_cap':
            if (entry.get('norm_batch') is not True or any(entry.get(key) is not False for key in
                    ('family_routing', 'attention_replay', 'attention_mask_once'))
                    or entry.get('replay_group_rows') != 4
                    or type(entry.get('lookup_max_rows')) is not int
                    or entry.get('lookup_max_rows') != (8 if entry[arm_key] else 32)):
                raise ValueError('Lookup cap comparison requires identical native attention and norm batching')
            if (entry['proposed'] != sum(block['rows'] - 1 for block in entry['blocks'])
                    or entry['accepted'] != sum(block['accepted'] for block in entry['blocks'])):
                raise ValueError('Lookup cap proposal accounting differs from recorded blocks')
        if arm_key == 'attention_replay' and (entry.get('norm_batch') is not True or entry.get('family_routing') is not True):
            raise ValueError('Both attention arms require identical norm batching and family routing')
        if arm_key == 'attention_wide':
            if any(entry.get(key) is not True for key in ('norm_batch', 'family_routing', 'attention_replay', 'attention_mask_once')):
                raise ValueError('Both width arms require identical replay, shared masks, norm batching and routing')
            width = entry.get('replay_group_rows')
            if type(width) is not int or width != (8 if entry[arm_key] else 4):
                raise ValueError('Width comparison requires four-row control and eight-row candidate')
        if any(entry[key] is not True for key in ('exact', 'state_exact', 'inactive_exact')):
            raise ValueError('Every request must pass native correctness')
        if any(entry[key] != reference[key] for key in identity):
            raise ValueError('Matched requests must use identical prompt, generation and proposal accounting')
        if arm_key == 'lookup_cap':
            arm_reference = next(record for record in requests if record[arm_key] is entry[arm_key])
            expected_blocks = [tuple(block[key] for key in routing) for block in arm_reference['blocks']]
        if [tuple(block[key] for key in routing) for block in entry['blocks']] != expected_blocks:
            raise ValueError('Matched requests changed proposal routing or acceptance')
        count = entry['committed_decode_tokens']
        if type(count) is not int or count <= 0 or count != len(entry['emitted']) - 1:
            raise ValueError('A nonzero post-seed request measurement is required')
        if sum(block['committed'] for block in entry['blocks']) != count:
            raise ValueError('Request block accounting differs from committed outputs')
        for key in ('decode_ms', 'engine_setup_ms', 'prefill_ms'):
            value = entry[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'decode_ms' and value == 0):
                raise ValueError('Finite measured request costs required')
        if entry['setup_amortized'] is not False or entry['cross_request_trace_reuse'] is not False:
            raise ValueError('Each arm requires its own request trace lifetime')
    arms = {}
    for name, enabled in (('control', False), ('candidate', True)):
        samples = [entry for entry in requests if entry[arm_key] is enabled]
        count = sum(entry['committed_decode_tokens'] for entry in samples)
        decode = sum(entry['decode_ms'] for entry in samples)
        setup = sum(entry['engine_setup_ms'] for entry in samples)
        prefill = sum(entry['prefill_ms'] for entry in samples)
        arms[name] = dict(requests=len(samples), committed=count, decode_ms=decode,
            engine_setup_ms=setup, prefill_ms=prefill, committed_tokens_per_second=1000 * count / decode,
            post_seed_including_setup_tokens_per_second=1000 * count / (setup + decode),
            prefill_setup_decode_ms=prefill + setup + decode)
    return dict(scope='One ABBA block of actual lookup-drafted requests; not coding-quality certification',
        arm_key=arm_key,
        arms=arms, decode_speedup=arms['control']['decode_ms'] / arms['candidate']['decode_ms'],
        setup_amortized=False, exact=True)


def measure_requests(measure, *, arm_key='norm_batch'):
    if arm_key not in ('norm_batch', 'attention_replay', 'attention_wide', 'lookup_cap', 'sampling_links'):
        raise ValueError('Known matched request experiment required')
    requests = [measure(**{arm_key: enabled}) for enabled in (False, True, True, False)]
    return requests, summarize_requests(requests, arm_key=arm_key)
