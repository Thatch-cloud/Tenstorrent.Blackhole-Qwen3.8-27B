"""Validate already-reserved offline KV pages; this does not allocate or reserve pages."""

from prefill_prefix_lookup import PrefixIdentity


class OfflinePrefixResidency:
    def __init__(self, operations, model, identity, reserved_pages):
        if (not isinstance(identity, PrefixIdentity) or not isinstance(reserved_pages, (tuple, list))
                or not reserved_pages or any(type(page) is not int or page < 0 for page in reserved_pages)
                or len(set(reserved_pages)) != len(reserved_pages)):
            raise ValueError('Explicit session identity and exclusively reserved physical pages required')
        self.operations, self.model, self.identity = operations, model, identity
        self.reserved = frozenset(reserved_pages)
        self.active = True
        self.bindings = self.current()
        self.capacity = min(record[0][0] for record in self.bindings)
        if max(self.reserved) >= self.capacity:
            raise ValueError('Reserved page exceeds native KV allocation')

    def current(self):
        caches = tuple(value for pair in self.model._paged_kv_caches for value in pair)
        if len(self.model._paged_kv_caches) != 16 or any(len(pair) != 2 for pair in self.model._paged_kv_caches):
            raise ValueError('All 16 native attention K/V pairs required')
        result = []
        for value in caches:
            shape = tuple(value.shape)
            parts = self.operations.get_device_tensors(value)
            if len(shape) != 4 or shape[2] != 64 or len(parts) != 2:
                raise ValueError('Both chip allocations with native 64-token KV pages required')
            result.append((shape, value.dtype, value.layout, value.memory_config(),
                tuple(part.buffer_address() for part in parts)))
        for chip in (0, 1):
            addresses = [record[-1][chip] for record in result]
            if len(set(addresses)) != len(addresses):
                raise ValueError('Native K/V layers must own distinct storage')
        return tuple(result)

    def invalidate(self):
        self.active = False

    def __call__(self, identity, pages, inactive_pages):
        try:
            if (not self.active or identity != self.identity or self.current() != self.bindings
                    or not isinstance(pages, (tuple, list)) or not pages
                    or not isinstance(inactive_pages, (tuple, list))
                    or any(type(page) is not int or not 0 <= page < self.capacity
                        for page in (*pages, *inactive_pages))
                    or len(set(pages)) != len(pages) or not set(pages).issubset(self.reserved)
                    or self.reserved.intersection(inactive_pages)):
                raise ValueError('Offline prefix KV reservation or native allocation changed')
        except BaseException:
            self.invalidate()
            raise
