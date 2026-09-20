"""Opt-in complete five-layer DFlash2 proposer with committed projected feature history."""

import hashlib
import os
from types import SimpleNamespace
from contextlib import nullcontext

from dflash_attention_mask import draft_attention_mask
from draft_attention_branch import prepare_attention_branch, execute_attention_branch
from draft_head_preparation import rope_tables
from draft_mlp_branch import prepare_mlp_branch, execute_mlp_branch
from draft_operation_audit import audit_operations
from draft_selector import select_active_candidates
from draft_shared_head import shared_head_candidates, merge_chunk_candidates
from feature_collective import gather_add_projection
from feature_projection import concatenate_local_features, projection_shards
from gdn_multitoken_conv import addresses, release_owned
from projection_link_policy import projection_links
from dflash_prefill_window import prefill_window


def pindiag(template, *values):
    """One loguru INFO line, brace-formatted, so the server log carries it; plain print
    where loguru is absent (host tests)."""
    try:
        from loguru import logger
    except ImportError:
        print(template.format(*values), flush=True)
        return
    logger.info(template, *values)


AUDIT_SWITCH = 'QWEN_FAST_PROPOSAL_AUDIT'
# The log capture truncates around 250 characters and loguru's own prefix takes
# some of them, so each audit message stays under this (the budget
# test_serving_sequential_step_shards holds the shard check to); a longer message
# is continued on further lines rather than lost.
AUDIT_LINE_BUDGET = 180
# Stages the eager proposal computes per chip by construction - each chip holds its
# own 16 query and four K/V heads (draft_attention_branch.py projection(), sharded
# dim 0), its own MLP columns (draft_mlp_branch.py device_projections), half the
# embedding width before the all-gather and half the vocabulary - so they have no
# cross-chip equality to check. Named once per proposal so their absence from the
# stage lines is not read as an omission.
AUDIT_SHARDED = ('embedding.local', 'attn q/k/v heads+output', 'o-proj partial', 'mlp gate/up/act',
                 'down-proj partial', 'vocab head top16')


def proposal_audit_enabled(environment=None):
    """QWEN_FAST_PROPOSAL_AUDIT=1. Read at each proposal rather than at import, so
    the switch a test flips is the one the proposal sees."""
    return (os.environ if environment is None else environment).get(AUDIT_SWITCH, '0') == '1'


def compare_shards(left, right):
    """One tensor's two chip copies, on the host: the bit-for-bit differing count
    (as serving_sequential_step.check_shards compares), the size of the difference,
    and which copy carries the larger norm."""
    import torch

    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        raise AssertionError('Chip shards differ in geometry: %s %s vs %s %s'
                             % (tuple(left.shape), left.dtype, tuple(right.shape), right.dtype))
    left, right = left.contiguous(), right.contiguous()
    bits = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}
    if left.dtype in bits:
        differing = left.view(bits[left.dtype]) != right.view(bits[left.dtype])
    else:
        differing = left != right
    wide_left, wide_right = left.to(torch.float64), right.to(torch.float64)
    difference = (wide_left - wide_right).abs()
    norms = (float(wide_left.norm()), float(wide_right.norm()))
    count = int(differing.sum())
    larger = None
    if count:
        larger = 'chip0' if norms[0] > norms[1] else 'chip1' if norms[1] > norms[0] else 'equal'
    return dict(differing=count, total=int(differing.numel()),
                max_abs=float(difference.max()) if count else 0.0,
                mean_abs=float(difference.mean()) if count else 0.0,
                finite=(bool(torch.isfinite(wide_left).all()), bool(torch.isfinite(wide_right).all())),
                norms=norms, larger_norm=larger)


def identity(value):
    return 'none' if value is None else '%s@%x' % (type(value).__name__, id(value))


def collective_state(collectives):
    """The host-side integer state of a collectives object, by attribute name: TT_CCL
    cycles its semaphore handles by index, so a shared object's position is on record."""
    try:
        attributes = vars(collectives)
    except TypeError:
        return 'opaque'
    state = {}
    for name, value in attributes.items():
        if type(value) is int or (isinstance(value, (list, tuple)) and value and all(type(item) is int for item in value)):
            state[name] = value
    return state or 'no-integer-attributes'


class ProposalAudit:
    """First-divergent-stage audit of one EAGER proposal, under QWEN_FAST_PROPOSAL_AUDIT=1.

    Runs 35486440095 and 35489404340: with every persistent buffer verified equal on
    both chips right before it, the second user's third eager proposal still made
    replicated selector features that differ between the chips - a transient of the
    proposal itself, after the other user's proposal had run. This names WHERE.
    Every replicated tensor the proposal reads or produces is read back from both
    chips at the stage that made it, compared bit for bit, and reported on one short
    '[AUDIT]' line: stage, shape, differing count, max_abs, mean_abs. The first stage
    that differs also says which chip's copy carries the larger norm. Per-chip
    intermediates (AUDIT_SHARDED) have no cross-chip equality and are named as such.

    At proposal start, the state this device reads but does not own is listed with
    id() and device addresses - the lent weights, the pool's histories, the
    collectives and its counters, the target model's tensors, the module-level state
    on the path - so a collision between two DFlashDevice instances shows as the same
    address in two devices' listings.

    Readbacks are synchronous and the audit is a diagnostic. Off, none of this runs:
    `observe` is never passed, and the proposal's operations and their order are
    exactly as before.
    """

    def __init__(self, device, *, log=None, line_budget=AUDIT_LINE_BUDGET):
        self.device, self.operations = device, device.operations
        self.log = pindiag if log is None else log
        self.line_budget = line_budget
        self.tag = 'dev=%x call=%d' % (id(device), device.proposal_calls)
        self.stages, self.first_divergent, self.summarized = [], None, False

    def line(self, template, *values):
        message = '[AUDIT] %s %s' % (self.tag, template.format(*values))
        budget = max(self.line_budget, 40)
        self.log('{}', message[:budget])
        rest = message[budget:]
        prefix = '[AUDIT] %s ...' % self.tag
        width = max(budget - len(prefix), 20)
        while rest:
            self.log('{}', prefix + rest[:width])
            rest = rest[width:]

    def address_of(self, value):
        try:
            return '%x/%x' % addresses(self.operations, value)
        except Exception:
            return 'n/a'

    def program_cache(self):
        count = getattr(getattr(self.device, 'mesh', None), 'num_program_cache_entries', None)
        if not callable(count):
            return 'n/a'
        try:
            return int(count())
        except Exception:
            return 'n/a'

    def begin(self):
        device = self.device
        self.line('begin position={} history_rows={} block_rows={} path=eager,uncached-history native_attention={} fused_convolution={}',
                  device.position, device.history_rows, device.block_rows,
                  getattr(device, 'native_proposal_attention', False), getattr(device, 'fused_convolution', False))
        self.line('sharded per chip, not compared: {}', ', '.join(AUDIT_SHARDED))
        self.not_owned()

    def not_owned(self):
        device = self.device
        weights = getattr(device, 'shared_weights', None)
        slot = getattr(device, 'pool_slot', None)
        collectives = getattr(device, 'collectives', None)
        model = getattr(device, 'model', None)
        kv = getattr(device, 'kv_history', None)
        self.line('not-owned weights={} borrowers={} model={} collectives={}', identity(weights),
                  len(getattr(weights, 'borrowers', ())), identity(model), identity(collectives))
        self.line('pool slot={} index={} owner={}', identity(slot), getattr(slot, 'index', None), getattr(slot, 'owner', None))
        self.line('collectives state={}', 'none' if collectives is None else collective_state(collectives))
        self.line('histories {}: history={} spare_history={}; kv banks {} (not read by the eager proposal)',
                  'lent by the pool' if slot is not None else 'owned', self.address_of(device.history),
                  self.address_of(device.spare_history), 'none' if kv is None else '%d layers' % len(getattr(kv, 'active', ())))
        embedding = getattr(model, 'embd', None)
        self.line('model tensors: lm_head_weight={} embd={} embd.weights={}',
                  self.address_of(getattr(model, 'lm_head_weight', None)), identity(embedding),
                  self.address_of(getattr(embedding, 'weights', None)))
        self.module_state()
        self.shared_weights(weights)

    def module_state(self):
        try:
            links = projection_links()
        except Exception as error:
            links = 'error:%s' % type(error).__name__
        try:
            from dflash_t16_native_scope import _ACTIVE

            admission = identity(_ACTIVE.get())
        except Exception:
            admission = 'unavailable'
        masks = getattr(self.device, 'validated_native_proposal_masks', ())
        self.line('module state: projection_links={} (lru_cache projection_link_policy._resolve) validated_masks={} '
                  'program_cache_entries={}', links, len(masks), self.program_cache())
        self.line('t16 admission={} (ContextVar dflash_t16_native_scope._ACTIVE)', admission)

    def shared_weights(self, weights):
        tensors = list(getattr(weights, 'tensors', None) or ())
        if not tensors:
            self.line('shared weights: none; the weights are this device\'s own uploads')
            return
        try:
            names = list(weights.names())
        except Exception:
            names = []
        if len(names) != len(tensors):
            names = ['tensor[%d]' % index for index in range(len(tensors))]
        entries = [(name, id(tensor), self.address_of(tensor)) for name, tensor in zip(names, tensors)]
        digest = hashlib.sha256(repr(entries).encode()).hexdigest()[:12]
        if digest == getattr(self.device, 'audit_digest', None):
            self.line('shared weights: {} tensors, ids and addresses unchanged since listed (digest {})', len(entries), digest)
            return
        self.device.audit_digest = digest
        self.line('shared weights: {} tensors, digest {}, listed as name id=<id()> addr=<chip0>/<chip1>', len(entries), digest)
        for name, tensor_id, address in entries:
            self.line('shared-tensor {} id={:x} addr={}', name, tensor_id, address)

    def observe(self, name, value):
        operations = self.operations
        shards = operations.get_device_tensors(value)
        if len(shards) != 2:
            raise AssertionError('Both chips required to audit %s' % name)
        left, right = (operations.to_torch(shard) for shard in shards)
        result = compare_shards(left, right)
        self.stages.append((name, result))
        marker = ''
        if not all(result['finite']):
            marker += ' finite=%d/%d' % tuple(int(flag) for flag in result['finite'])
        if result['differing'] and self.first_divergent is None:
            self.first_divergent = name
            marker += ' FIRST larger_norm=%s' % result['larger_norm']
        self.line('stage={} shape={} differing={} of {} max_abs={:g} mean_abs={:g}{}',
                  name, 'x'.join(str(extent) for extent in left.shape), result['differing'], result['total'],
                  result['max_abs'], result['mean_abs'], marker)
        return value

    def summary(self):
        if self.summarized:
            return
        self.summarized = True
        diverged = [name for name, result in self.stages if result['differing']]
        first = next((result for name, result in self.stages if name == self.first_divergent), None)
        self.line('summary stages={} diverged={} first_divergent={}{} program_cache_entries={}',
                  len(self.stages), len(diverged), self.first_divergent or 'none',
                  '' if first is None else ' larger_norm=%s norms=%g/%g' % (first['larger_norm'], *first['norms']),
                  self.program_cache())


class PreparedDraftWeights:
    """Every device tensor a DFlashDevice reads as a weight - the five learned layers'
    attention and MLP parameters, the feature projection and its norm, the final norm
    and the selector projection - uploaded once.

    A device without a shared set prepares its own, through its own `retain`, in
    exactly the order the constructor always used. On the serving path one set is
    prepared at attach and lent to every device, because these are the FIRST buffers
    a request allocates, and a request admitted after another request's verify trace
    was captured allocates them into the addresses that trace baked for the
    intermediates it freed: first-fit hands the lowest freed hole to the first
    allocation. Every later replay of that trace then writes over the weights, per
    chip, and the two replicas of a replicated weight no longer agree - which is
    what runs 35477522469 and 35479238722 measured in the selector projection, at
    the second-admitted user's second proposal. Prepared once, before any request,
    the set predates every request trace (serving_buffer_pool.py has the full
    account; precedent docs/experiment-execution.md, feature_prefix.py).
    """

    def __init__(self, operations, mesh, layers, projection, selector, *, block_rows, live_query_qk=False,
                 native_proposal_attention=False, retain=None):
        import torch

        layers = tuple(layers)
        if (len(layers) != 5 or type(block_rows) is not int or block_rows not in (8, 16, 32)
                or type(live_query_qk) is not bool or type(native_proposal_attention) is not bool
                or (retain is not None and not callable(retain))):
            raise ValueError('All five DFlash2 layers and an explicit proposal geometry required')
        self.operations, self.mesh, self.block_rows = operations, mesh, block_rows
        self.live_query_qk, self.native_proposal_attention = live_query_qk, native_proposal_attention
        self.sources = (layers, projection, selector)
        self.owned, self.tensors, self.borrowers = [], [], []
        self.closed = False
        keep = self.retain if retain is None else retain

        def own(value):
            self.tensors.append(value)
            return keep(value)

        def upload(value, *, sharded=False, row_major=False):
            return own(operations.from_torch(value, device=mesh, dtype=operations.bfloat16,
                layout=operations.ROW_MAJOR_LAYOUT if row_major else operations.TILE_LAYOUT,
                memory_config=operations.DRAM_MEMORY_CONFIG,
                mesh_mapper=operations.ShardTensorToMesh(mesh, dim=0) if sharded else operations.ReplicateTensorToMesh(mesh)))

        try:
            self.layers = []
            for attention, convolution, mlp in layers:
                self.layers.append((prepare_attention_branch(operations, mesh, attention, convolution, own,
                    native_head_layout=True, block_rows=block_rows, live_query_qk=live_query_qk,
                    **(dict(native_proposal_attention=True) if native_proposal_attention else {})),
                    prepare_mlp_branch(operations, mesh, mlp, convolution, own), mlp, convolution))
            shards = projection_shards(projection['fc.weight'])
            self.projection = upload(torch.cat(shards, dim=0), sharded=True)
            self.feature_norm = upload(projection['hidden_norm.weight'].reshape(1, 1, 160, 32), row_major=True)
            self.final_norm = upload(selector['norm.weight'].reshape(1, 1, 160, 32), row_major=True)
            self.selector_projection = upload(selector['candidate_selector.hidden_projection.weight'].T.contiguous())
            self.predecessors = selector['candidate_selector.predecessor_codebook'].double()
            self.successors = selector['candidate_selector.successor_codebook'].double()
        except BaseException:
            self.close()
            raise

    def retain(self, value):
        self.owned.append(value)
        return value

    def names(self):
        """A name per uploaded tensor, in upload order, from where each sits in the
        prepared dicts - so a diverged address in a shard check can be placed."""
        located = {}

        def visit(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit('%s.%s' % (prefix, key), item)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit('%s[%d]' % (prefix, index), item)
            elif any(value is tensor for tensor in self.tensors):
                located.setdefault(id(value), prefix)

        for index, (attention, mlp, _, _) in enumerate(self.layers):
            visit('layer%d.attention' % index, {key: item for key, item in attention.items()
                                                if key not in ('operations', 'mesh', 'source_weights', 'source_convolution')})
            visit('layer%d.mlp' % index, {key: item for key, item in mlp.items()
                                          if key not in ('operations', 'mesh', 'source_weights', 'source_convolution')})
        for name in ('projection', 'feature_norm', 'final_norm', 'selector_projection'):
            visit(name, getattr(self, name))
        return [located.get(id(tensor), 'unnamed[%d]' % index) for index, tensor in enumerate(self.tensors)]

    def describe(self):
        return dict(tensors=len(self.tensors), borrowers=[getattr(borrower, 'name', str(borrower)) for borrower in self.borrowers],
            weights=[dict(name=name, addresses=list(addresses(self.operations, tensor)))
                     for name, tensor in zip(self.names(), self.tensors, strict=True)])

    def lend(self, borrower, *, mesh, layers, projection, selector, block_rows, live_query_qk, native_proposal_attention):
        layers = tuple(layers)
        if self.closed:
            raise ValueError('Closed shared draft weights cannot be lent')
        if (mesh is not self.mesh or block_rows != self.block_rows or live_query_qk != self.live_query_qk
                or native_proposal_attention != self.native_proposal_attention):
            raise ValueError('Shared draft weights were prepared for another mesh or proposal geometry')
        shared_layers, shared_projection, shared_selector = self.sources
        if (len(layers) != len(shared_layers) or projection is not shared_projection or selector is not shared_selector
                or any(len(layer) != 3 or any(mine is not theirs for mine, theirs in zip(shared, layer))
                       for shared, layer in zip(shared_layers, layers))):
            raise ValueError('Shared draft weights were prepared from other learned layers')
        if any(borrower is other for other in self.borrowers):
            raise ValueError('Shared draft weights are already lent to this borrower')
        self.borrowers.append(borrower)
        pindiag('[PINDIAG] draft weights lent to {} (borrowers={} tensors={})',
                getattr(borrower, 'name', borrower), len(self.borrowers), len(self.tensors))
        return self

    def release(self, borrower):
        index = next((position for position, other in enumerate(self.borrowers) if other is borrower), None)
        if index is None:
            raise ValueError('Shared draft weights were not lent to this borrower')
        del self.borrowers[index]
        pindiag('[PINDIAG] draft weights returned by {} (borrowers={})',
                getattr(borrower, 'name', borrower), len(self.borrowers))

    def close(self):
        if self.closed:
            return
        self.closed = True
        borrowers = [getattr(borrower, 'name', borrower) for borrower in self.borrowers]
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.tensors.clear()
        self.borrowers.clear()
        if borrowers:
            raise ValueError('Shared draft weights closed while lent to %r' % borrowers)


class DFlashDevice:
    def __init__(self, operations, model, collectives, layers, projection, selector, features, *, position, progress=None,
                 block_rows=8, proposal_capture=False, max_new_tokens=513, fused_convolution=False, feature_start=0,
                 cache_history=False, cache_projection_capture=False, live_query_qk=False, native_proposal_attention=False,
                 defer_proposal_capture=False, buffer_pool=None, shared_weights=None):
        import torch

        window = prefill_window(position)
        features = tuple(features)
        if (model.num_devices != 2 or model.vocab_size != 248320 or not model._lmhead_vocab_sharded
                or len(layers) != 5 or type(feature_start) is not int or feature_start != window['start']
                or len(features) != 5 or any(len(value.shape) != 4 or value.shape[2] != window['rows'] for value in features)
                or type(block_rows) is not int or block_rows not in (8, 16, 32) or type(proposal_capture) is not bool
                or type(fused_convolution) is not bool or type(cache_history) is not bool
                or type(defer_proposal_capture) is not bool or (defer_proposal_capture and not proposal_capture)
                or (cache_history and (not proposal_capture or block_rows not in (8, 16)))
                or type(cache_projection_capture) is not bool or (cache_projection_capture and not cache_history)
                or type(live_query_qk) is not bool or (live_query_qk and (not proposal_capture or block_rows != 8))
                or type(native_proposal_attention) is not bool or (native_proposal_attention and
                    (not proposal_capture or not cache_history or block_rows not in (8, 16) or live_query_qk or cache_projection_capture))
                or (buffer_pool is not None and not callable(getattr(buffer_pool, 'acquire', None)))
                or (shared_weights is not None and not callable(getattr(shared_weights, 'lend', None)))):
            raise ValueError('Pinned TP2 target, all five DFlash2 layers and bounded prefill required')
        self.operations, self.model, self.mesh, self.collectives = operations, model, model.mesh_device, collectives
        self.position, self.history_rows = position, window['rows']
        self.block_rows, self.max_drafts = block_rows, block_rows - 1
        self.owned, self.layers = [], []
        # Device tensors this device reads but does not own: never freed here, and
        # protected from every temporary the same way the owned ones are.
        self.borrowed = []
        self.name = 'DFlashDevice@%x position=%d' % (id(self), position)
        self.history = self.pending = None
        self.spare_history = None
        self.pool_slot = self.shared_weights = None
        self.proposal_capture = None
        self.kv_history = None
        self.cache_history = cache_history
        self.live_query_qk, self.validated_live_masks = live_query_qk, set()
        self.native_proposal_attention, self.validated_native_proposal_masks = native_proposal_attention, set()
        self.fused_convolution, self.convolution_checks = fused_convolution, []
        self.closed = False
        self.proposal_calls = self.published_rows = 0
        # Digest of the shared weights' ids and addresses as the proposal audit last
        # listed them, so an unchanged set is one line rather than the listing again.
        self.audit_digest = None
        if progress is not None and not callable(progress):
            raise ValueError('An optional callable audit progress reporter is required')
        self.progress = progress
        self.kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        try:
            if buffer_pool is not None:
                # First, so an exhausted pool refuses the request before any upload.
                self.pool_slot = buffer_pool.acquire(owner=self.name)
            geometry = dict(block_rows=block_rows, live_query_qk=live_query_qk,
                            native_proposal_attention=native_proposal_attention)
            if shared_weights is None:
                weights = PreparedDraftWeights(operations, self.mesh, layers, projection, selector,
                                               retain=self.retain, **geometry)
            else:
                # Borrowed, not uploaded: the weights are the first thing a request
                # allocates, so they take the lowest hole an earlier request's verify
                # trace left behind, and its replays overwrite them (PreparedDraftWeights).
                weights = shared_weights.lend(self, mesh=self.mesh, layers=layers, projection=projection,
                                              selector=selector, **geometry)
                self.shared_weights = weights
                self.borrowed.extend(weights.tensors)
            self.layers = weights.layers
            self.projection, self.feature_norm = weights.projection, weights.feature_norm
            self.final_norm, self.selector_projection = weights.final_norm, weights.selector_projection
            self.predecessors, self.successors = weights.predecessors, weights.successors
            self.history = self.project_features(features, self.history_rows)
            padded = operations.pad(self.history, [(0, 0), (0, 0), (0, 2048 - self.history_rows), (0, 0)], 0.0)
            if addresses(operations, padded) != addresses(operations, self.history):
                operations.deallocate(self.history)
            self.history = padded
            if self.pool_slot is None:
                self.spare_history = operations.zeros_like(self.history)
            else:
                # Borrowed, not allocated. Storage allocated here - after another
                # request's verify trace was captured - can sit at an address that
                # trace baked for an intermediate it freed, and every replay then
                # writes over it, per chip, so the replicas diverge (runs 35477522469,
                # 35479238722; serving_buffer_pool.py). The pool's pair predates
                # every request trace.
                operations.copy(self.history, self.pool_slot.history)
                operations.deallocate(self.history)
                self.history, self.spare_history = self.pool_slot.history, self.pool_slot.spare_history
            operations.synchronize_device(self.mesh)
            if cache_history:
                from draft_kv_history import DraftKVHistory

                # Pooled, the cache adopts the slot's K/V banks rather than allocating
                # its own: run 35481466425 found kv_history[0].k of the other request
                # overwritten by a step while its pooled history survived. The slot's
                # zero query goes with them when the pool carries one: it is read at
                # every proposal for the request's life and was the last per-request
                # upload the draft cache made after another request's traces existed.
                self.kv_history = DraftKVHistory(operations, self.mesh, [layer[0] for layer in self.layers], self.history,
                    position=position, history_rows=self.history_rows, capture_projection=cache_projection_capture,
                    **(dict(storage=self.pool_slot.kv) if self.pool_slot is not None else {}),
                    **(dict(query=self.pool_slot.query) if getattr(self.pool_slot, 'query', None) is not None else {}))
                if self.progress is not None:
                    self.kv_history.audit(self.history)
            if proposal_capture and not defer_proposal_capture:
                from dflash_proposal_trace import PreparedDFlashProposal

                self.proposal_capture = PreparedDFlashProposal(self, max_new_tokens=max_new_tokens)
        except BaseException:
            self.close()
            raise

    def retain(self, value):
        self.owned.append(value)
        return value

    def upload(self, value, *, sharded=False, row_major=False):
        operations = self.operations
        return self.retain(operations.from_torch(value, device=self.mesh, dtype=operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT if row_major else operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(self.mesh, dim=0) if sharded else operations.ReplicateTensorToMesh(self.mesh)))

    def temporaries(self, protected):
        owned = []
        # getattr, as execute_proposal reads its flags: fixtures stand a bare namespace in for the device.
        protected_ids = [addresses(self.operations, value) for value in [*protected, *getattr(self, 'borrowed', ())]]
        def retain(value):
            identity = addresses(self.operations, value)
            if identity not in protected_ids:
                if any(any(left == right for left, right in zip(identity, other, strict=True)) for other in protected_ids):
                    raise ValueError('Draft temporary must not partially alias protected storage')
                owned.append(value)
            return value
        return owned, retain

    def release_except(self, owned, output):
        identity = addresses(self.operations, output) if output is not None else None
        release_owned(self.operations, [value for value in owned if addresses(self.operations, value) != identity])

    def project_features(self, features, count, row_offset=None):
        # A packed block's taps hold every user's rows (packed_verifier.PackedFeatureTaps
        # names this user's first row); unpacked taps start at row 0.
        offset = getattr(features, 'row_offset', 0) if row_offset is None else row_offset
        features = tuple(features)
        operations = self.operations
        if (len(features) != 5 or type(count) is not int or count < 1 or type(offset) is not int or offset < 0
                or any(len(value.shape) != 4 or tuple(value.shape)[:2] != (1, 1)
                    or value.shape[2] < offset + count or value.shape[3] != 2560 or value.dtype != operations.bfloat16
                    for value in features)):
            raise ValueError('Five complete ordered BF16 local feature taps required')
        owned, retain = self.temporaries(features)
        output = None
        try:
            chunks = []
            program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 10),
                in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=2,
                fuse_batch=True, fused_activation=None, mcast_in0=True)
            for start in range(0, count, 32):
                rows = min(32, count - start)
                parts = []
                for value in features:
                    sliced = retain(operations.slice(value, (0, 0, offset + start, 0), (1, 1, offset + start + rows, 2560)))
                    if rows < 32:
                        sliced = retain(operations.pad(sliced, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0))
                    parts.append(sliced)
                joined = retain(concatenate_local_features(operations, parts))
                partial = retain(operations.matmul(joined, self.projection, dtype=operations.float32,
                    compute_kernel_config=self.kernel, program_config=program, memory_config=operations.DRAM_MEMORY_CONFIG))
                summed = retain(gather_add_projection(operations, self.mesh, self.collectives, partial, retain_temporaries=retain))
                rounded = retain(operations.typecast(summed, operations.bfloat16))
                normalized = retain(operations.rms_norm(rounded, epsilon=1e-6, weight=self.feature_norm,
                    compute_kernel_config=self.kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
                chunks.append(retain(operations.slice(normalized, (0, 0, 0, 0), (1, 1, rows, 5120))))
            output = retain(operations.concat(chunks, dim=2)) if len(chunks) > 1 else chunks[0]
            operations.synchronize_device(self.mesh)
        except BaseException:
            release_owned(operations, owned)
            raise
        self.release_except(owned, output)
        return output

    def prepare_publication(self, features, prefix, *, position):
        if self.closed or self.pending is not None or position != self.position or type(prefix) is not int or not 1 <= prefix <= 32:
            raise ValueError('One live target-feature publication at the committed frontier required')
        operations = self.operations
        owned, retain = self.temporaries([self.history, self.spare_history])
        output = None
        cache_publication = None
        try:
            projected = retain(self.project_features(features, prefix))
            valid_history = retain(operations.slice(self.history, (0, 0, 0, 0), (1, 1, self.history_rows, 5120)))
            combined = retain(operations.concat([valid_history, projected], dim=2))
            rows = min(2048, self.history_rows + prefix)
            output = retain(operations.slice(combined, (0, 0, combined.shape[2] - rows, 0), (1, 1, combined.shape[2], 5120)))
            padded = retain(operations.pad(output, [(0, 0), (0, 0), (0, 2048 - rows), (0, 0)], 0.0))
            operations.copy(padded, self.spare_history)
            if self.kv_history is not None:
                cache_publication = self.kv_history.prepare(projected, prefix, position=position)
            operations.synchronize_device(self.mesh)
            self.pending = SimpleNamespace(position=position, prefix=prefix, rows=rows, history=self.spare_history,
                kv=cache_publication, status='prepared')
        except BaseException:
            if cache_publication is not None:
                self.kv_history.discard(cache_publication)
            release_owned(operations, owned)
            raise
        release_owned(operations, owned)
        return self.pending

    def commit_publication(self, publication):
        if self.closed or publication is not self.pending or publication.status != 'prepared' or publication.position != self.position:
            raise ValueError('Only the current prepared feature publication may commit')
        if self.kv_history is not None:
            self.kv_history.commit(publication.kv)
        previous = self.history
        self.history, self.history_rows = publication.history, publication.rows
        self.spare_history = previous
        self.position += publication.prefix
        self.published_rows += publication.prefix
        publication.status = 'committed'
        self.pending = None
        if self.kv_history is not None and self.progress is not None:
            self.kv_history.audit(self.history)

    def discard_publication(self, publication):
        if publication.status == 'committed':
            return
        if publication is not self.pending or publication.status != 'prepared':
            raise ValueError('Only the current prepared feature publication may be discarded')
        if self.kv_history is not None:
            self.kv_history.discard(publication.kv)
        publication.status = 'discarded'
        self.pending = None

    def execute_proposal(self, identifiers, history, mask, rope, *, context, owned, retain, stage, audit=True,
                         audit_convolution=False, cached_history=None, pack=None, observe=None):
        operations = self.operations
        live_query_qk = getattr(self, 'live_query_qk', False)
        native_proposal_attention = getattr(self, 'native_proposal_attention', False)
        if live_query_qk and addresses(operations, mask) not in self.validated_live_masks:
            raise ValueError('Live-query proposal mask was not validated before upload/capture')
        if native_proposal_attention and addresses(operations, mask) not in self.validated_native_proposal_masks:
            raise ValueError('Native proposal mask was not validated before upload/capture')
        if cached_history is not None and (self.kv_history is None or len(cached_history) != len(self.layers)):
            raise ValueError('Every prepared learned layer requires a committed K/V cache')
        if type(audit_convolution) is not bool or (audit_convolution and not self.fused_convolution):
            raise ValueError('Convolution audit requires the explicit fused candidate')
        # After the guards, so an unregistered mask or a bad audit request still fails
        # on its own terms rather than on a missing attribute.
        # Packed, every one of the 32 rows is a live proposal, so there is nothing to
        # pad away and nothing to trim off before the vocabulary head; and the causal
        # convolution must restart at each user's first row.
        rows = 32 if pack is not None else self.block_rows
        seams = None if pack is None else tuple(
            (index * self.block_rows, (index + 1) * self.block_rows) for index in range(len(pack)))
        # QWEN_FAST_PROPOSAL_AUDIT (ProposalAudit): `observe(name, tensor)` reads a
        # replicated intermediate back from both chips at the stage that made it. None,
        # the default and the only value with the audit off, adds no operation and
        # changes no order; each branch gets a copy scoped to its layer.
        def watch(name, value):
            if observe is not None:
                observe(name, value)
            return value

        def scoped(prefix):
            if observe is None:
                return {}
            return dict(observe=lambda name, value: observe('%s.%s' % (prefix, name), value))

        stage('borrowed-embedding')
        local = retain(self.model.embd(identifiers, memory_config=operations.DRAM_MEMORY_CONFIG))
        local = retain(operations.reshape(local, (1, 1, rows, 2560)))
        stage('embedding-all-gather')
        hidden = retain(operations.experimental.all_gather_async(local, persistent_output_buffer=None, dim=3,
            multi_device_global_semaphore=self.collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.collectives.get_and_cycle_barrier_semaphore_handle(), num_links=projection_links(),
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2))
        watch('embedding.gathered', hidden)
        if rows != 32:
            hidden = watch('embedding.padded', retain(operations.pad(hidden, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)))
        for layer, (attention, mlp, weights, convolution) in enumerate(self.layers):
            convolution_options = {}
            if getattr(self, 'fused_convolution', False):
                from draft_convolution_fused import checked_convolution

                def convolve(*args, **kwargs):
                    return checked_convolution(*args, **kwargs, audit=audit_convolution, checks=self.convolution_checks,
                        context=dict(position=self.position, layer=layer))
                convolution_options['convolution_operation'] = convolve
            stage('attention', layer=layer)
            operation_audit = audit_operations(operations, self.mesh, self.progress) if audit and self.progress is not None and layer == 0 and self.proposal_calls else nullcontext()
            with operation_audit:
                hidden = execute_attention_branch(operations, self.mesh, self.collectives, hidden, history, mask, rope,
                    retain, parameters=attention, context=context, pack=pack, **convolution_options,
                    **scoped('layer%d.attn' % layer),
                    **(dict(live_query_mask_validated=True) if live_query_qk else {}),
                    **(dict(native_proposal_mask_validated=True) if native_proposal_attention else {}),
                    **(dict(cached_history=[cache[layer] for cache in cached_history] if pack is not None
                    else cached_history[layer]) if cached_history is not None else {}))
            stage('mlp', layer=layer)
            hidden = execute_mlp_branch(operations, self.mesh, self.collectives, hidden, weights, convolution,
                retain, parameters=mlp, trace_safe=True, **convolution_options, **scoped('layer%d.mlp' % layer),
                **(dict(boundaries=seams) if pack is not None else {}))['output']
        stage('final-norm-and-selector-projection')
        normalized = watch('selector.normalized', retain(operations.rms_norm(hidden, epsilon=1e-6, weight=self.final_norm,
            compute_kernel_config=self.kernel, memory_config=operations.DRAM_MEMORY_CONFIG)))
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 1),
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        projected = watch('selector.projected-fp32', retain(operations.matmul(normalized, self.selector_projection, dtype=operations.float32,
            program_config=program, compute_kernel_config=self.kernel, memory_config=operations.DRAM_MEMORY_CONFIG)))
        projected = watch('selector.features', retain(operations.typecast(projected, operations.bfloat16)))
        block = watch('head.input', retain(operations.slice(normalized, (0, 0, 0, 0), (1, 1, rows, 5120))))
        stage('shared-full-vocabulary-head')
        chunks = shared_head_candidates(operations, self.model, block, owned)
        return SimpleNamespace(projected=projected, chunks=chunks)

    def proposal_snapshot(self, outputs):
        tensors = [outputs.projected, *(chunk[name] for chunk in outputs.chunks for name in ('values', 'indices'))]
        return tuple(self.operations.to_torch(shard).clone() for tensor in tensors
            for shard in self.operations.get_device_tensors(tensor))

    def select_proposal(self, outputs, seed, count):
        import torch

        operations = self.operations
        host_chunks = []
        for chunk in outputs.chunks:
            values = operations.get_device_tensors(chunk['values'])
            indices = operations.get_device_tensors(chunk['indices'])
            if len(values) != 2 or len(indices) != 2:
                raise AssertionError('Both learned head shards required')
            for chip in range(2):
                host_chunks.append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                    values=operations.to_torch(values[chip]).float().reshape(self.block_rows, 16),
                    indices=operations.to_torch(indices[chip]).long().reshape(self.block_rows, 16)))
        candidates, unary = merge_chunk_candidates(host_chunks, block_rows=self.block_rows)
        projected_parts = [operations.to_torch(value) for value in operations.get_device_tensors(outputs.projected)]
        if len(projected_parts) != 2 or not torch.equal(*projected_parts):
            # Runs 35478872085 and 35479238722 both died here on the third block of
            # two users, and neither the per-request trace nor per-request
            # collectives explained it. Report the SIZE and SHAPE of the divergence:
            # a tiny difference is fidelity, a large one is memory, and all-finite
            # versus not separates a bad read from a bad write.
            detail = 'shards=%d' % len(projected_parts)
            if len(projected_parts) == 2:
                left, right = (value.float() for value in projected_parts)
                difference = (left - right).abs()
                detail = ('call=%d position=%d rows=%d max_abs=%g mean_abs=%g '
                          'differing=%d of %d finite=%s/%s'
                          % (self.proposal_calls, self.position, self.history_rows,
                             float(difference.max()), float(difference.mean()),
                             int((difference > 0).sum()), difference.numel(),
                             bool(torch.isfinite(left).all()), bool(torch.isfinite(right).all())))
            raise AssertionError('Replicated learned selector features differ: %s' % detail)
        selector_hidden = projected_parts[0][..., 1:self.block_rows, :].reshape(1, self.max_drafts, 256)
        tokens, unused = select_active_candidates(selector_hidden, candidates, unary, self.predecessors, self.successors,
            torch.tensor([seed], dtype=torch.int64))
        return tuple(int(token) for token in tokens[0, :count])

    def propose(self, seed, count):
        import torch

        if self.closed or self.pending is not None or type(seed) is not int or not 0 <= seed < 248320 or type(count) is not int or not 1 <= count <= self.max_drafts:
            raise ValueError('Committed DFlash2 history and bounded anchor/proposal IDs required')
        if self.proposal_capture is not None:
            if proposal_audit_enabled():
                pindiag('[AUDIT] dev={:x} call={} proposal is a trace replay: the stage audit reads intermediates '
                        'only from the eager proposal (QWEN_FAST_EAGER_PROPOSAL=1)', id(self), self.proposal_calls)
            tokens = self.proposal_capture.propose(seed, count)
            self.proposal_calls += 1
            return tokens
        operations = self.operations
        audit = ProposalAudit(self) if proposal_audit_enabled() else None
        owned, retain = self.temporaries([self.history, self.spare_history, *self.owned])
        previous_stage = 'target-publication'
        def stage(name, **values):
            nonlocal previous_stage
            if self.progress is not None:
                self.progress('draft-fence', after=previous_stage, next_step=name, position=self.position)
                operations.synchronize_device(self.mesh)
                self.progress('draft-step', step=name, position=self.position, **values)
            previous_stage = name
        def upload(value, dtype=operations.bfloat16, row_major=False):
            return retain(operations.from_torch(value, device=self.mesh, dtype=dtype,
                layout=operations.ROW_MAJOR_LAYOUT if row_major else operations.TILE_LAYOUT,
                memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh)))
        try:
            if audit is not None:
                audit.begin()
            stage('upload-anchor-and-mask')
            identifiers = upload(torch.tensor([[seed, *([248070] * self.max_drafts)]], dtype=torch.int64), dtype=operations.uint32, row_major=True)
            if audit is not None:
                audit.observe('input.identifiers', identifiers)
            stage('prepare-history-mask-and-rope')
            host_mask = draft_attention_mask(self.history_rows, block_rows=self.block_rows)
            key_rows = host_mask.shape[-1]
            valid_history = retain(operations.slice(self.history, (0, 0, 0, 0), (1, 1, self.history_rows, 5120)))
            history = retain(operations.pad(valid_history, [(0, 0), (0, 0), (0, key_rows - self.history_rows), (0, 0)], 0.0))
            mask = upload(host_mask)
            if self.native_proposal_attention:
                # The TRACE path validates and registers its mask; the eager path
                # never did, so it was refused by its own device with 'Native
                # proposal mask was not validated before upload/capture'
                # (run 35478076344). Same check, same registration.
                if self.block_rows == 16:
                    from dflash_t16_native_scope import require_active
                    from dflash_t16_native_attention import validate_mask

                    require_active()
                else:
                    from proposal_native_attention import validate_mask

                validate_mask(host_mask)
                self.validated_native_proposal_masks.add(addresses(operations, mask))
            rope = {name: tuple(upload(value) for value in rope_tables(start, rows))
                for name, start, rows in (('q', self.position, 32), ('k', self.position - self.history_rows, key_rows))}
            if audit is not None:
                # The inputs as the proposal consumes them: the committed buffer, the
                # window sliced and padded from it, the mask and the position tables.
                audit.observe('input.history-buffer', self.history)
                audit.observe('input.history-window', history)
                audit.observe('input.mask', mask)
                for name in ('q', 'k'):
                    for part, table in zip(('cos', 'sin'), rope[name]):
                        audit.observe('input.rope.%s.%s' % (name, part), table)
            outputs = self.execute_proposal(identifiers, history, mask, rope, context=self.history_rows,
                owned=owned, retain=retain, stage=stage, **(dict(observe=audit.observe) if audit is not None else {}))
            if audit is not None:
                audit.summary()
            stage('synchronize-head-and-selector')
            operations.synchronize_device(self.mesh)
            stage('read-candidates-and-select')
            tokens = self.select_proposal(outputs, seed, count)
            self.proposal_calls += 1
            return tokens
        finally:
            if audit is not None:
                # Already written on the normal path; after a failure inside
                # execute_proposal this is where the stages seen so far are summed up.
                audit.summary()
            stage('synchronize-and-release-draft-temporaries')
            operations.synchronize_device(self.mesh)
            release_owned(operations, owned)

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        if self.proposal_capture is not None:
            self.proposal_capture.close()
        if self.pending is not None:
            self.discard_publication(self.pending)
        if self.kv_history is not None:
            self.kv_history.close()
        if self.pool_slot is not None:
            # Commits swap the pair; whichever way round, both belong to the pool.
            for name in ('history', 'spare_history'):
                if any(getattr(self, name) is value for value in self.pool_slot.tensors):
                    setattr(self, name, None)
            self.pool_slot.release()
            self.pool_slot = None
        if self.shared_weights is not None:
            self.shared_weights.release(self)
            self.shared_weights = None
        self.borrowed.clear()
        if self.history is not None:
            self.operations.deallocate(self.history)
            self.history = None
        if self.spare_history is not None:
            self.operations.deallocate(self.spare_history)
            self.spare_history = None
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True
