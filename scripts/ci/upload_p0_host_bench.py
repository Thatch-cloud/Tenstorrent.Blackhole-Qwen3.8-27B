"""Host-side cost of what the upload levers remove, measured on a CPU (no card, no ttnn): the zero buffers' host build and the weights' host transpose.

    python3 scripts/ci/upload_p0_host_bench.py [--threads N] [--repeat N] [--blocks N] [--layers N]

It prints one JSON object. What it measures and what it does not:
  * kv_zero_host: `torch.zeros` of one KV tensor at the served shape plus ONE full read pass over it, which is the least the host bf8 packer must do to those bytes. The
    packer itself (tt-metal's C++ pack) and the PCIe write are not here: those were measured on the cards (engine_start_phases: the kv_pool phase). With
    QWEN_FAST_DEVICE_ZEROS=1 this host work is zero.
  * pool_zero_host: the host zero tensors ServingBufferPool.allocate builds for N users, counted by running the pool against a recording fake (bytes the lever leaves on
    the host, bytes it moves to the card). The pool's other 2 GB are device-side clones and uploads of other kinds.
  * shard_w_transpose: `w.to(bf16).T.contiguous()` for the MLP weights (w1, w3: [17408, 5120], w2: [5120, 17408]), the copy tp_common.shard_w makes before as_tensor looks at
    its cache; times and GB/s, and the total for the model's layers. With QWEN_FAST_LAZY_SHARD_W=1 a cache hit does none of it.
The numbers move with the host's load (this machine serves other work): the minimum of --repeat runs is reported, with the 1-minute load average beside it.
"""

import argparse
import json
import os
import sys
import time

import torch

HIDDEN, INTERMEDIATE, LAYERS = 5120, 17408, 64
KV_SHAPE = (19968, 1, 64, 256)          # one tensor of the 8 x 262k pool: blocks, local kv heads, block size, head dim


def best(function, repeat):
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return min(times)


def kv_zero_host(shape, repeat):
    def build():
        zeros = torch.zeros(shape, dtype=torch.bfloat16)
        zeros.view(torch.int16).max()                       # one pass over every element: what any packer has to do
    seconds = best(build, repeat)
    count = 1
    for extent in shape:
        count *= extent
    return dict(shape=list(shape), host_bytes=count * 2, seconds=round(seconds, 4), tensors_per_pool=32, pool_seconds=round(32 * seconds, 2))


def shard_w_transpose(repeat, layers):
    out = {}
    total = 0.0
    for name, rows, columns, per_layer in (('w1_w3', INTERMEDIATE, HIDDEN, 2), ('w2', HIDDEN, INTERMEDIATE, 1)):
        weight = torch.zeros((rows, columns), dtype=torch.bfloat16)
        weight.view(torch.int16).add_(1)                    # touch the pages: a freshly mapped zero page would make the copy cheap
        seconds = best(lambda: weight.to(torch.bfloat16).T.contiguous(), repeat)
        size = rows * columns * 2
        out[name] = dict(shape=[rows, columns], bytes=size, seconds=round(seconds, 4), gb_per_second=round(size / seconds / 1e9, 2))
        total += seconds * per_layer * layers
    out['mlp_total_seconds'] = round(total, 1)
    out['mlp_total_bytes'] = layers * 3 * INTERMEDIATE * HIDDEN * 2
    return out


def pool_zero_host(users):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_serving_buffer_pool import FakeOperations
    import serving_buffer_pool

    class Counting(FakeOperations):
        bfloat8_b, float32 = 'bf8', 'fp32'

        def __init__(self):
            FakeOperations.__init__(self)
            self.host_bytes = self.device_bytes = 0

        def from_torch(self, value, **options):
            self.host_bytes += value.numel() * value.element_size()
            return FakeOperations.from_torch(self, value, **options)

        def empty(self, shape, *, dtype, layout, device, memory_config):
            count = 1
            for extent in shape:
                count *= extent
            self.device_bytes += count * 2
            return self.allocate(tuple(shape), dtype=dtype, layout=layout, mapper=('replicate', device), memory=memory_config)

    results = {}
    for label, flag in (('off', '0'), ('on', '1')):
        os.environ['QWEN_FAST_DEVICE_ZEROS'] = flag
        import qwen_device_zeros
        qwen_device_zeros.forget()
        operations = Counting()
        serving_buffer_pool.ServingBufferPool(operations, 'mesh', users=users)
        results[label] = dict(host_zero_bytes=operations.host_bytes, device_filled_bytes=operations.device_bytes)
    os.environ.pop('QWEN_FAST_DEVICE_ZEROS', None)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--threads', type=int, default=1, help='torch threads (the host packer and the transpose are single threaded; default 1)')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--blocks', type=int, default=KV_SHAPE[0])
    parser.add_argument('--layers', type=int, default=LAYERS)
    parser.add_argument('--users', type=int, default=8)
    arguments = parser.parse_args(argv)
    torch.set_num_threads(arguments.threads)
    shape = (arguments.blocks,) + KV_SHAPE[1:]
    report = dict(load_average_1m=round(os.getloadavg()[0], 2), threads=arguments.threads, repeat=arguments.repeat,
                  kv_zero_host=kv_zero_host(shape, arguments.repeat),
                  shard_w_transpose=shard_w_transpose(arguments.repeat, arguments.layers),
                  pool_zero_host=pool_zero_host(arguments.users))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
