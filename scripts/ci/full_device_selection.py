"""Paired verifier plus argmax/readback costs; excludes drafting and dynamic publication."""

import time

from attention_batch import capture_operation
from force_argmax import sample_rows
from model_batch import ModelBatch


def measure_selection(model, sampler, prompt, oracle, pages, helpers, checkpoints, initial, *, rows,
                      prefill, decode, save, restore, live_digest, kv_digest, local_host):
    import torch
    import ttnn

    if rows not in (1, 2, 4, 8, 16, 32) or len(oracle) < rows:
        raise ValueError('Complete supported native-reference block required')
    mesh, length = model.mesh_device, len(prompt)
    prefill(prompt)
    expected = torch.cat([decode(oracle[index], length + index, False).reshape(-1, model.args.vocab_size)[:1]
                          for index in range(rows)], dim=0)
    expected_ids = expected.float().argmax(dim=-1)
    expected_state, expected_kv = live_digest(), kv_digest(length + rows)
    prefill(prompt)
    save(initial)
    fixtures, captured, outputs = {}, {}, {}
    setup_started = time.perf_counter()

    def operation(arm):
        logits = fixtures[arm].run(sharded_logits=arm == 'device')
        try:
            ids = sample_rows(sampler, logits, rows, ttnn) if arm == 'device' else None
            return logits, ids
        except BaseException:
            ttnn.deallocate(logits)
            raise

    def release(result):
        for value in result:
            if value is not None:
                ttnn.deallocate(value)

    def read_ids(arm, result, *, both=True):
        tensor = result[0] if arm == 'host' else result[1]
        values = local_host(tensor) if both else [ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])]
        if arm == 'host':
            return [value.reshape(rows, model.args.vocab_size).float().argmax(dim=-1) for value in values]
        return [value.reshape(-1)[:rows].to(torch.long) for value in values]

    def validate(arm, result):
        values = local_host(result[0])
        if len(values) != 2:
            raise AssertionError('Both vocabulary shards required')
        full = [torch.cat(values, dim=-1)] if arm == 'device' else values
        if any(not torch.equal(value.reshape(rows, model.args.vocab_size), expected) for value in full):
            raise AssertionError('Pre-gather logits differ from native full-vocabulary reference')
        predictions = read_ids(arm, result)
        if len(predictions) != 2 or any(not torch.equal(value, expected_ids) for value in predictions):
            raise AssertionError('Device force-argmax differs from native greedy selection')
        if live_digest() != expected_state or kv_digest(length + rows) != expected_kv:
            raise AssertionError('Selection arm changed native GDN/KV state')

    try:
        for arm in ('host', 'device'):
            fixtures[arm] = ModelBatch(model, oracle[:rows], length, pages, helpers, checkpoints, rows,
                serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
                batch_conv=True, packed_checkpoints=True, ordered_cache=True)
            restore(initial)
            result = operation(arm)
            try:
                validate(arm, result)
            finally:
                release(result)
        for arm in ('host', 'device'):
            restore(initial)
            captured[arm], outputs[arm] = capture_operation(ttnn, mesh, lambda arm=arm: operation(arm))
            restore(initial)
            ttnn.execute_trace(mesh, captured[arm], cq_id=0, blocking=True)
            validate(arm, outputs[arm])
        ttnn.synchronize_device(mesh)
        setup_ms = (time.perf_counter() - setup_started) * 1000
        samples = []
        for block in range(3):
            for arm in ('host', 'device', 'device', 'host'):
                elapsed = []
                for repeat in range(10):
                    restore(initial)
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                    ttnn.execute_trace(mesh, captured[arm], cq_id=0, blocking=True)
                    predictions = read_ids(arm, outputs[arm], both=False)
                    elapsed.append((time.perf_counter() - started) * 1000)
                    if len(predictions) != 1 or any(not torch.equal(value, expected_ids) for value in predictions):
                        raise AssertionError('Timed greedy output changed')
                validate(arm, outputs[arm])
                samples.append(dict(block=block, arm=arm, replay_ms=elapsed, mean_ms=sum(elapsed) / len(elapsed)))
        return dict(length=length, rows=rows, exact=True, setup_ms=setup_ms, samples=samples,
            timed_readback='First chip only in both arms; both chips independently checked outside timing',
            scope='Paired captured verifier plus argmax/readback; restore and setup excluded; no drafting or dynamic commit')
    finally:
        for trace in captured.values():
            ttnn.release_trace(mesh, trace)
        for result in outputs.values():
            release(result)
        for fixture in fixtures.values():
            fixture.close()
