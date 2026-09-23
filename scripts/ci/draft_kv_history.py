"""Double-buffered projected draft history with explicit accepted-prefix publication."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

from draft_head_preparation import rope_tables
from draft_kv_projection import project_key_value
from gdn_multitoken_conv import addresses, release_owned


KV_SHAPE = (1, 4, 2048, 128)
QUERY_SHAPE = (1, 1, 32, 2048)


def validate_query(operations, query):
    """The pre-trace zero query input lent by the serving pool, or None to upload one here.

    The projection only reads it (draft_kv_projection.project_key_value, through
    draft_head_layout.split_projected_heads), but it is read at every proposal and every
    publication for the request's whole life - so, uploaded here after an earlier
    request's traces exist, it is one more buffer their replays can overwrite.
    """
    if query is None:
        return None
    if tuple(getattr(query, 'shape', ())) != QUERY_SHAPE or getattr(query, 'dtype', None) != operations.bfloat16:
        raise ValueError('A replicated zero BF16 (1, 1, 32, 2048) query is required')
    return query


def validate_storage(operations, storage, layers):
    """Pre-trace K/V banks lent by the serving pool - per learned layer an active and a
    spare bank of k and v, handed over zeroed - or None to allocate the banks here.

    Allocated here, the banks are the buffers run 35481466425 found overwritten by the
    other request's verify replay while that request's pooled history survived: they
    were allocated after the replaying trace existed, into the holes it had baked
    (serving_buffer_pool.py). Lent from the pool they predate every request trace.
    """
    if storage is None:
        return None
    banks = tuple(storage)
    if (len(banks) != layers or any(not isinstance(bank, dict) or set(bank) != {'active', 'spare'}
            or any(not isinstance(bank[side], dict) or set(bank[side]) != {'k', 'v'}
                   or any(tuple(value.shape) != KV_SHAPE or value.dtype != operations.bfloat16
                          for value in bank[side].values())
                   for side in ('active', 'spare')) for bank in banks)):
        raise ValueError('One active and one spare replicated BF16 (1, 4, 2048, 128) K/V bank per learned layer required')
    identities = [addresses(operations, value) for value in bank_tensors(banks)]
    if any(any(left == right for left, right in zip(identity, other, strict=True))
           for index, identity in enumerate(identities) for other in identities[:index]):
        raise ValueError('Lent K/V banks must own independent chip storage')
    return banks


def bank_tensors(banks):
    return [bank[side][head] for bank in banks for side in ('active', 'spare') for head in ('k', 'v')]


@contextmanager
def indexed_temporaries(cache, protected):
    """DraftKVHistory.temporaries under QWEN_FAST_ROUND_B1 (C2): the same scope - the
    same values kept, refused and queued, released in the same order at the same point -
    with set lookups instead of list scans, and the lent banks' and query's addresses
    computed once per borrowed set and kept with the tensors (a lent tensor stays at one
    address while it is lent).

    The scan it replaces keeps a value whose identity is protected, raises when any one
    chip's address matches the same chip of any protected identity, and otherwise queues
    the value; on exit it releases the queued values whose identity is not protected and
    not one of the cache's own owned tensors, recomputed from cache.owned THEN (the
    constructor adds to it inside a scope). addresses() gives one address per chip, so
    "some chip matches" is "that chip's address is in that chip's set".

    QWEN_FAST_ROUND_B1_AUDIT re-reads the cached borrowed addresses on every reuse, makes
    every retain() decision again by the list scan and every release list again by the
    list filter, and compares (dflash_packed_proposal.ROUND_B1_AUDIT_FLAG)."""
    audit = os.environ.get('QWEN_FAST_ROUND_B1_AUDIT') == '1'
    operations = cache.operations
    owned = []
    projection_owned = cache.projection.owned if cache.projection is not None else []
    identities = [addresses(operations, value) for value in [*cache.owned, *projection_owned]]
    borrowed = tuple(cache.borrowed)
    cached = getattr(cache, '_round_b1_borrowed', None)
    if (cached is None or len(cached[0]) != len(borrowed)
            or any(mine is not theirs for mine, theirs in zip(cached[0], borrowed))):
        cached = (borrowed, [addresses(operations, value) for value in borrowed])
        cache._round_b1_borrowed = cached
    elif audit and borrowed:
        from dflash_packed_proposal import audit_borrowed

        audit_borrowed([addresses(operations, value) for value in borrowed], cached[1])
    identities.extend(cached[1])
    identities.extend(addresses(operations, value) for value in protected)
    exact = set(identities)
    chips = [set() for _ in range(max((len(identity) for identity in identities), default=0))]
    for identity in identities:
        for chip, address in enumerate(identity):
            chips[chip].add(address)

    def retain(value):
        identity = addresses(operations, value)
        if identity not in exact:
            if any(address in known for address, known in zip(identity, chips)):
                raise ValueError('Draft cache temporary partially aliases borrowed storage')
            owned.append(value)
        return value
    if audit:
        from dflash_packed_proposal import audit_release, audited_retain

        scoped = audited_retain(retain, owned, lambda value: addresses(operations, value), identities)
    else:
        scoped = retain
    try:
        yield scoped
    finally:
        persistent = set(exact)
        persistent.update(addresses(operations, value) for value in cache.owned)
        released = [value for value in owned if addresses(operations, value) not in persistent]
        if audit:
            listed = [*identities, *(addresses(operations, value) for value in cache.owned)]
            expected = [value for value in owned if addresses(operations, value) not in listed]
        release_owned(operations, released)
        if audit:
            audit_release(expected, released)


class DraftKVHistory:
    def __init__(self, operations, mesh, parameters, features, *, position, history_rows, capture_projection=False,
                 storage=None, query=None):
        import torch

        parameters = tuple(parameters)
        if (type(position) is not int or not 1 <= position <= 262112
                or type(history_rows) is not int or history_rows != min(position, 2048)
                or not 1 <= len(parameters) <= 5 or type(capture_projection) is not bool):
            raise ValueError('Bounded absolute draft frontier and explicit learned layers required')
        banks = validate_storage(operations, storage, len(parameters))
        query = validate_query(operations, query)
        self.operations, self.mesh, self.parameters = operations, mesh, parameters
        self.position, self.history_rows = position, history_rows
        self.owned, self.active, self.spare = [], [], []
        # Lent banks and query: read and written here, protected from every temporary
        # like the owned tensors are, never freed here.
        self.borrowed = [] if banks is None else bank_tensors(banks)
        if query is not None:
            self.borrowed.append(query)
        self.checks = []
        self.projection = None
        self.pending, self.closed = None, False
        try:
            if query is None:
                self.query = self.upload(torch.zeros(QUERY_SHAPE, dtype=torch.bfloat16))
                self.owned.append(self.query)
            else:
                # Lent zeroed (serving_buffer_pool.py), exactly as the upload it replaces.
                self.query = query
            with self.temporaries([features]) as retain:
                inputs, tables = self.project_inputs(features, history_rows, position - history_rows, retain)
                for layer, parameter in enumerate(parameters):
                    result = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                    active, spare = {}, {}
                    for name in ('k', 'v'):
                        valid = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, history_rows, 128)))
                        if banks is None:
                            active[name] = retain(operations.pad(valid, [(0, 0), (0, 0), (0, 2048 - history_rows), (0, 0)], 0.0))
                            self.owned.append(active[name])
                            spare[name] = operations.zeros_like(active[name])
                            self.owned.append(spare[name])
                            continue
                        # Into the lent active bank; the padded source goes with the scope.
                        operations.copy(retain(operations.pad(valid, [(0, 0), (0, 0), (0, 2048 - history_rows), (0, 0)], 0.0)),
                                        banks[layer]['active'][name])
                        active[name], spare[name] = banks[layer]['active'][name], banks[layer]['spare'][name]
                    self.active.append(active)
                    self.spare.append(spare)
                if banks is not None:
                    # Before the scope frees the padded sources the copies read from.
                    operations.synchronize_device(mesh)
            operations.synchronize_device(mesh)
            if capture_projection:
                from draft_kv_projection_trace import PreparedDraftKVProjection

                self.projection = PreparedDraftKVProjection(operations, mesh, parameters, self.query)
        except BaseException:
            self.close()
            raise

    def upload(self, value):
        operations = self.operations
        return operations.from_torch(value, device=self.mesh, dtype=operations.bfloat16,
            layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))

    @contextmanager
    def temporaries(self, protected):
        if os.environ.get('QWEN_FAST_ROUND_B1') == '1':
            with indexed_temporaries(self, protected) as retain:
                yield retain
            return
        owned = []
        projection_owned = self.projection.owned if self.projection is not None else []
        identities = [addresses(self.operations, value)
                      for value in [*self.owned, *projection_owned, *self.borrowed, *protected]]
        def retain(value):
            identity = addresses(self.operations, value)
            if identity not in identities:
                if any(any(left == right for left, right in zip(identity, other, strict=True)) for other in identities):
                    raise ValueError('Draft cache temporary partially aliases borrowed storage')
                owned.append(value)
            return value
        try:
            yield retain
        finally:
            persistent = [*identities, *(addresses(self.operations, value) for value in self.owned)]
            release_owned(self.operations, [value for value in owned if addresses(self.operations, value) not in persistent])

    def project_inputs(self, features, count, start, retain):
        operations = self.operations
        if (type(count) is not int or not 1 <= count <= 2048 or len(features.shape) != 4
                or tuple(features.shape)[:2] != (1, 1) or features.shape[2] < count
                or features.shape[3] != 5120 or features.dtype != operations.bfloat16):
            raise ValueError('Complete replicated projected BF16 feature rows required')
        padded_rows = ((count + 31) // 32) * 32
        valid = retain(operations.slice(features, (0, 0, 0, 0), (1, 1, count, 5120)))
        inputs = retain(operations.pad(valid, [(0, 0), (0, 0), (0, padded_rows - count), (0, 0)], 0.0))
        host = rope_tables(start, padded_rows)
        for table in host:
            table[..., count:, :] = 0
        tables = tuple(retain(self.upload(table)) for table in host)
        return inputs, tables

    def prepare(self, features, prefix, *, position):
        if (self.closed or self.pending is not None or type(position) is not int or position != self.position
                or type(prefix) is not int or not 1 <= prefix <= 32 or position + prefix > 262144):
            raise ValueError('One accepted-prefix cache update at the current committed frontier required')
        operations = self.operations
        rows = min(2048, self.history_rows + prefix)
        with self.temporaries([features]) as retain:
            inputs, tables = self.project_inputs(features, prefix, position, retain)
            projected = self.projection.project(inputs, tables) if self.projection is not None else None
            for layer, (parameter, active, spare) in enumerate(zip(self.parameters, self.active, self.spare, strict=True)):
                result = projected[layer] if projected is not None else project_key_value(
                    operations, inputs, self.query, tables, retain, parameters=parameter)
                for name in ('k', 'v'):
                    historical = retain(operations.slice(active[name], (0, 0, 0, 0), (1, 4, self.history_rows, 128)))
                    accepted = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, prefix, 128)))
                    combined = retain(operations.concat([historical, accepted], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
                    tail = retain(operations.slice(combined, (0, 0, self.history_rows + prefix - rows, 0),
                        (1, 4, self.history_rows + prefix, 128)))
                    padded = retain(operations.pad(tail, [(0, 0), (0, 0), (0, 2048 - rows), (0, 0)], 0.0))
                    operations.copy(padded, spare[name])
            operations.synchronize_device(self.mesh)
        self.pending = SimpleNamespace(position=position, prefix=prefix, rows=rows, status='prepared')
        return self.pending

    def commit(self, publication):
        if (self.closed or publication is not self.pending or publication.status != 'prepared'
                or publication.position != self.position):
            raise ValueError('Only the current prepared draft cache may commit')
        self.active, self.spare = self.spare, self.active
        self.position += publication.prefix
        self.history_rows = publication.rows
        publication.status = 'committed'
        self.pending = None

    def discard(self, publication):
        if publication.status == 'committed':
            return
        if self.closed or publication is not self.pending or publication.status != 'prepared':
            raise ValueError('Only the current prepared draft cache may be discarded')
        publication.status = 'discarded'
        self.pending = None

    def audit(self, features):
        import torch

        if self.closed or self.pending is not None:
            raise ValueError('Only a committed open draft cache may be audited')
        operations = self.operations
        with self.temporaries([features]) as retain:
            inputs, tables = self.project_inputs(features, self.history_rows, self.position - self.history_rows, retain)
            for layer, (parameter, active) in enumerate(zip(self.parameters, self.active, strict=True)):
                expected = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                for name in ('k', 'v'):
                    actual_shards = operations.get_device_tensors(active[name])
                    expected_shards = operations.get_device_tensors(expected[name])
                    if len(actual_shards) != 2 or len(expected_shards) != 2:
                        raise AssertionError('Both committed draft-cache shards required')
                    for chip, (actual, reference) in enumerate(zip(actual_shards, expected_shards, strict=True)):
                        left = operations.to_torch(actual)[..., :self.history_rows, :].contiguous()
                        right = operations.to_torch(reference)[..., :self.history_rows, :].contiguous()
                        if not torch.equal(left.view(torch.int16), right.view(torch.int16)):
                            raise AssertionError(f'Committed historical K/V differs: layer={layer}, head={name}, chip={chip}')
                        self.checks.append(dict(position=self.position, rows=self.history_rows, layer=layer, head=name, chip=chip, exact=True))

    def close(self):
        if self.closed:
            return
        if self.pending is not None:
            self.discard(self.pending)
        self.operations.synchronize_device(self.mesh)
        if self.projection is not None:
            self.projection.close()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.active.clear()
        self.spare.clear()
        self.borrowed.clear()
        self.closed = True
