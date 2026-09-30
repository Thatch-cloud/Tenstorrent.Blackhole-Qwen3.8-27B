"""draft_kv_history.DraftKVHistory at any served width: the drafter's double-buffered projected K/V history.

draft_kv_history.py must stay byte for byte what the frozen bundle carries - draft_kv_slide_adapter and dflash_traced_publish
text-patch DraftKVHistory.prepare's exact source at attach, and its (1, 4, ...) slices are that text - so the four-card
class is a sibling, not an edit. The drafter is TP-sharded: a chip holds 4 KV heads at the pair and 2 at four cards
(tp_shapes.draft_kv_heads), and the zero query input is 32 rows of the chip's query heads (2048 wide, 1024). This class is
the pair's with those numbers from tp_shapes; everything else (the scope managers, project_inputs, commit, discard, close)
is inherited. dflash_device builds this class instead of the pair's at four cards. The slide adapter and the traced
publish's K/V fusion are the pair's text patches and are not selected at four cards (the four-card profiles turn their
flags off).
"""

from types import SimpleNamespace

import draft_kv_history as pair
import draft_kv_slide_tp
from draft_kv_projection import project_key_value
import tp_shapes
from tp_addresses import addresses


def kv_shape():
    return (1, tp_shapes.active().draft_kv_heads, 2048, 128)


def query_shape():
    return (1, 1, 32, tp_shapes.active().draft_query)


def validate_query(operations, query):
    """The pre-trace zero query input lent by the serving pool at this width, or None to upload one here."""
    if query is None:
        return None
    if tuple(getattr(query, 'shape', ())) != query_shape() or getattr(query, 'dtype', None) != operations.bfloat16:
        raise ValueError('A replicated zero BF16 %s query is required' % (query_shape(),))
    return query


def validate_storage(operations, storage, layers):
    """Pre-trace K/V banks lent by the serving pool at this width (active and spare k / v per learned layer), or None."""
    if storage is None:
        return None
    banks = tuple(storage)
    shape = kv_shape()
    if (len(banks) != layers or any(not isinstance(bank, dict) or set(bank) != {'active', 'spare'}
            or any(not isinstance(bank[side], dict) or set(bank[side]) != {'k', 'v'}
                   or any(tuple(value.shape) != shape or value.dtype != operations.bfloat16
                          for value in bank[side].values())
                   for side in ('active', 'spare')) for bank in banks)):
        raise ValueError('One active and one spare replicated BF16 %s K/V bank per learned layer required' % (shape,))
    identities = [addresses(operations, value) for value in pair.bank_tensors(banks)]
    if any(any(left == right for left, right in zip(identity, other, strict=True))
           for index, identity in enumerate(identities) for other in identities[:index]):
        raise ValueError('Lent K/V banks must own independent chip storage')
    return banks


class DraftKVHistory(pair.DraftKVHistory):
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
        kv_heads = tp_shapes.active().draft_kv_heads
        self.operations, self.mesh, self.parameters = operations, mesh, parameters
        self.position, self.history_rows = position, history_rows
        self.owned, self.active, self.spare = [], [], []
        self.borrowed = [] if banks is None else pair.bank_tensors(banks)
        if query is not None:
            self.borrowed.append(query)
        self.checks = []
        self.projection = None
        self.pending, self.closed = None, False
        try:
            if query is None:
                self.query = self.upload(torch.zeros(query_shape(), dtype=torch.bfloat16))
                self.owned.append(self.query)
            else:
                self.query = query
            with self.temporaries([features]) as retain:
                inputs, tables = self.project_inputs(features, history_rows, position - history_rows, retain)
                for layer, parameter in enumerate(parameters):
                    result = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                    active, spare = {}, {}
                    for name in ('k', 'v'):
                        valid = retain(operations.slice(result[name], (0, 0, 0, 0), (1, kv_heads, history_rows, 128)))
                        if banks is None:
                            active[name] = retain(operations.pad(valid, [(0, 0), (0, 0), (0, 2048 - history_rows), (0, 0)], 0.0))
                            self.owned.append(active[name])
                            spare[name] = operations.zeros_like(active[name])
                            self.owned.append(spare[name])
                            continue
                        operations.copy(retain(operations.pad(valid, [(0, 0), (0, 0), (0, 2048 - history_rows), (0, 0)], 0.0)),
                                        banks[layer]['active'][name])
                        active[name], spare[name] = banks[layer]['active'][name], banks[layer]['spare'][name]
                    self.active.append(active)
                    self.spare.append(spare)
                if banks is not None:
                    operations.synchronize_device(mesh)
            operations.synchronize_device(mesh)
            if capture_projection:
                from draft_kv_projection_trace import PreparedDraftKVProjection

                self.projection = PreparedDraftKVProjection(operations, mesh, parameters, self.query)
        except BaseException:
            self.close()
            raise

    def prepare(self, features, prefix, *, position):
        if (self.closed or self.pending is not None or type(position) is not int or position != self.position
                or type(prefix) is not int or not 1 <= prefix <= 32 or position + prefix > 262144):
            raise ValueError('One accepted-prefix cache update at the current committed frontier required')
        operations = self.operations
        kv_heads = tp_shapes.active().draft_kv_heads
        rows = min(2048, self.history_rows + prefix)
        slide = draft_kv_slide_tp.enabled()   # QWEN_FAST_TP_KV_SLIDE; unset keeps the eager chain
        with self.temporaries([features]) as retain:
            inputs, tables = self.project_inputs(features, prefix, position, retain)
            projected = self.projection.project(inputs, tables) if self.projection is not None else None
            for layer, (parameter, active, spare) in enumerate(zip(self.parameters, self.active, self.spare, strict=True)):
                result = projected[layer] if projected is not None else project_key_value(
                    operations, inputs, self.query, tables, retain, parameters=parameter)
                for name in ('k', 'v'):
                    if slide:
                        # One generic_op over the fixed-shape banks with history_rows, prefix, drop and rows as runtime
                        # arguments: no program is built per (history_rows, prefix), which the eager chain below does in the ramp.
                        draft_kv_slide_tp.prepare(self.mesh, active[name], result[name], spare[name],
                                                  history_rows=self.history_rows, prefix=prefix)()
                        continue
                    historical = retain(operations.slice(active[name], (0, 0, 0, 0), (1, kv_heads, self.history_rows, 128)))
                    accepted = retain(operations.slice(result[name], (0, 0, 0, 0), (1, kv_heads, prefix, 128)))
                    combined = retain(operations.concat([historical, accepted], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
                    tail = retain(operations.slice(combined, (0, 0, self.history_rows + prefix - rows, 0),
                        (1, kv_heads, self.history_rows + prefix, 128)))
                    padded = retain(operations.pad(tail, [(0, 0), (0, 0), (0, 2048 - rows), (0, 0)], 0.0))
                    operations.copy(padded, spare[name])
            operations.synchronize_device(self.mesh)
        self.pending = SimpleNamespace(position=position, prefix=prefix, rows=rows, status='prepared')
        return self.pending

    def audit(self, features):
        import torch

        if self.closed or self.pending is not None:
            raise ValueError('Only a committed open draft cache may be audited')
        operations = self.operations
        chips = tp_shapes.chip_count()
        with self.temporaries([features]) as retain:
            inputs, tables = self.project_inputs(features, self.history_rows, self.position - self.history_rows, retain)
            for layer, (parameter, active) in enumerate(zip(self.parameters, self.active, strict=True)):
                expected = project_key_value(operations, inputs, self.query, tables, retain, parameters=parameter)
                for name in ('k', 'v'):
                    actual_shards = operations.get_device_tensors(active[name])
                    expected_shards = operations.get_device_tensors(expected[name])
                    if len(actual_shards) != chips or len(expected_shards) != chips:
                        raise AssertionError('%s committed draft-cache shards required' % tp_shapes.all_chips())
                    for chip, (actual, reference) in enumerate(zip(actual_shards, expected_shards, strict=True)):
                        left = operations.to_torch(actual)[..., :self.history_rows, :].contiguous()
                        right = operations.to_torch(reference)[..., :self.history_rows, :].contiguous()
                        if not torch.equal(left.view(torch.int16), right.view(torch.int16)):
                            raise AssertionError(f'Committed historical K/V differs: layer={layer}, head={name}, chip={chip}')
                        self.checks.append(dict(position=self.position, rows=self.history_rows, layer=layer, head=name,
                                                chip=chip, exact=True))
