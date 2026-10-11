"""The engine's side of a COLD tier for evicted prefix blocks: an abstract client interface and an in-memory fake (a stub; nothing in the engine calls it yet).

The host KV tier (qwen_prefix_registry.TierStore) keeps the attention-KV pages vLLM evicted from the device pool in host RAM. A cold tier would hold the pages that
leave host RAM too, on storage a service owns. What the engine needs from such a service is small and is all that is pinned here, so that the service can be built
against it and the engine's side tested without one:

  * a block is named by a ColdKey: the TENANT id (the opaque tag of the salt, qwen_prefix_registry.tenant_of_salt: a block of one tenant can never be named with
    another's), the LAYOUT version (a string that changes whenever the bytes of a block would mean something else: the image, the model, the block size and bytes, the
    tile layout - the tier's fingerprint is the engine's source for it), and the block hash (vLLM's chained block hash, which already carries the salt);
  * fetch(key, out) writes the block's bytes into `out`, a writable buffer the CALLER owns (the engine passes a slot of its pinned host slab, so no copy is made and no
    allocation happens on the path) and returns a Future of the number of bytes written: 0 for a block the service does not hold, an error for a buffer too small;
  * store(key, data) hands the bytes of a block to the service and returns a Future of the number of bytes it holds for the key;
  * contains(key) is a Future of whether the service holds the key;
  * every call is asynchronous: it returns at once, the result (or the failure) is the Future's, and the engine never blocks the scheduler on one;
  * a key with another layout than the client's is an error (ColdLayoutError), never a miss and never bytes of another layout.

The service itself - where it runs, how it stores, how a tenant's bytes are kept apart and accounted - is not this module's business and not this repository's.
Stdlib only.
"""

import abc
import threading
from collections import namedtuple
from concurrent.futures import Future, ThreadPoolExecutor

ColdKey = namedtuple('ColdKey', ('tenant', 'layout', 'block_hash'))


class ColdTierError(RuntimeError):
    """The service could not do what was asked (an unreachable service, a refused request)."""


class ColdLayoutError(ColdTierError):
    """The key names a layout version the client does not serve."""


class ColdBufferError(ColdTierError):
    """The caller's buffer is smaller than the block."""


class ColdTierClient(abc.ABC):
    """The interface (the module docstring). `layout` is the layout version this client serves."""

    layout = ''

    @abc.abstractmethod
    def fetch(self, key, out):
        """Write the block named by `key` into the writable buffer `out`. -> Future of the bytes written (0: absent)."""

    @abc.abstractmethod
    def store(self, key, data):
        """Hand the block's bytes to the service. -> Future of the bytes held for the key."""

    @abc.abstractmethod
    def contains(self, key):
        """-> Future of whether the service holds the key."""

    def close(self):
        """Release what the client holds. The default holds nothing."""


class InMemoryColdTier(ColdTierClient):
    """A fake service: a dict behind a small thread pool, so that every call is really asynchronous. `delay` is called in the worker before each operation (a test's
    gate or sleep); `fail(operation, key)` returning True makes that operation fail with ColdTierError."""

    def __init__(self, layout, delay=None, fail=None, workers=2):
        self.layout = layout
        self.delay = delay
        self.fail = fail
        self.blocks = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cold-tier-fake')

    def _submit(self, operation, key, work):
        future = Future()
        if key.layout != self.layout:
            future.set_exception(ColdLayoutError('layout %r is not the %r this client serves' % (key.layout, self.layout)))
            return future

        def run():
            try:
                if self.delay is not None:
                    self.delay()
                if self.fail is not None and self.fail(operation, key):
                    raise ColdTierError('%s failed (injected)' % operation)
                future.set_result(work())
            except BaseException as error:        # noqa: BLE001 - the Future carries it
                future.set_exception(error)

        self.executor.submit(run)
        return future

    def fetch(self, key, out):
        def work():
            with self.lock:
                data = self.blocks.get(key)
            if data is None:
                return 0
            view = memoryview(out).cast('B')
            if len(view) < len(data):
                raise ColdBufferError('the buffer holds %d bytes, the block is %d' % (len(view), len(data)))
            view[:len(data)] = data
            return len(data)

        return self._submit('fetch', key, work)

    def store(self, key, data):
        payload = bytes(data)

        def work():
            with self.lock:
                self.blocks[key] = payload
            return len(payload)

        return self._submit('store', key, work)

    def contains(self, key):
        def work():
            with self.lock:
                return key in self.blocks

        return self._submit('contains', key, work)

    def close(self):
        self.executor.shutdown(wait=True)
