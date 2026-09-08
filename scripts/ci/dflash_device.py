"""Opt-in complete five-layer DFlash2 proposer with committed projected feature history."""

from types import SimpleNamespace
from contextlib import nullcontext

from draft_attention import draft_attention_mask
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


class DFlashDevice:
    def __init__(self, operations, model, collectives, layers, projection, selector, features, *, position, progress=None,
                 block_rows=8, proposal_capture=False, max_new_tokens=513, fused_convolution=False, feature_start=0,
                 cache_history=False):
        import torch

        window = prefill_window(position)
        features = tuple(features)
        if (model.num_devices != 2 or model.vocab_size != 248320 or not model._lmhead_vocab_sharded
                or len(layers) != 5 or type(feature_start) is not int or feature_start != window['start']
                or len(features) != 5 or any(len(value.shape) != 4 or value.shape[2] != window['rows'] for value in features)
                or type(block_rows) is not int or block_rows not in (8, 32) or type(proposal_capture) is not bool
                or type(fused_convolution) is not bool or type(cache_history) is not bool
                or (cache_history and (not proposal_capture or block_rows != 8))):
            raise ValueError('Pinned TP2 target, all five DFlash2 layers and bounded prefill required')
        self.operations, self.model, self.mesh, self.collectives = operations, model, model.mesh_device, collectives
        self.position, self.history_rows = position, window['rows']
        self.block_rows, self.max_drafts = block_rows, block_rows - 1
        self.owned, self.layers = [], []
        self.history = self.pending = None
        self.spare_history = None
        self.proposal_capture = None
        self.kv_history = None
        self.cache_history = cache_history
        self.fused_convolution, self.convolution_checks = fused_convolution, []
        self.closed = False
        self.proposal_calls = self.published_rows = 0
        if progress is not None and not callable(progress):
            raise ValueError('An optional callable audit progress reporter is required')
        self.progress = progress
        self.kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        try:
            for attention, convolution, mlp in layers:
                self.layers.append((prepare_attention_branch(operations, self.mesh, attention, convolution, self.retain,
                    native_head_layout=True, block_rows=block_rows),
                    prepare_mlp_branch(operations, self.mesh, mlp, convolution, self.retain), mlp, convolution))
            shards = projection_shards(projection['fc.weight'])
            self.projection = self.upload(torch.cat(shards, dim=0), sharded=True)
            self.feature_norm = self.upload(projection['hidden_norm.weight'].reshape(1, 1, 160, 32), row_major=True)
            self.final_norm = self.upload(selector['norm.weight'].reshape(1, 1, 160, 32), row_major=True)
            self.selector_projection = self.upload(selector['candidate_selector.hidden_projection.weight'].T.contiguous())
            self.predecessors = selector['candidate_selector.predecessor_codebook'].double()
            self.successors = selector['candidate_selector.successor_codebook'].double()
            self.history = self.project_features(features, self.history_rows)
            padded = operations.pad(self.history, [(0, 0), (0, 0), (0, 2048 - self.history_rows), (0, 0)], 0.0)
            if addresses(operations, padded) != addresses(operations, self.history):
                operations.deallocate(self.history)
            self.history = padded
            self.spare_history = operations.zeros_like(self.history)
            operations.synchronize_device(self.mesh)
            if cache_history:
                from draft_kv_history import DraftKVHistory

                self.kv_history = DraftKVHistory(operations, self.mesh, [layer[0] for layer in self.layers], self.history,
                    position=position, history_rows=self.history_rows)
                if self.progress is not None:
                    self.kv_history.audit(self.history)
            if proposal_capture:
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
        protected_ids = [addresses(self.operations, value) for value in protected]
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

    def project_features(self, features, count):
        features = tuple(features)
        operations = self.operations
        if (len(features) != 5 or type(count) is not int or count < 1
                or any(len(value.shape) != 4 or tuple(value.shape)[:2] != (1, 1)
                    or value.shape[2] < count or value.shape[3] != 2560 or value.dtype != operations.bfloat16 for value in features)):
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
                    sliced = retain(operations.slice(value, (0, 0, start, 0), (1, 1, start + rows, 2560)))
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
                         audit_convolution=False, cached_history=None):
        operations = self.operations
        if cached_history is not None and (self.kv_history is None or len(cached_history) != len(self.layers)):
            raise ValueError('Every prepared learned layer requires a committed K/V cache')
        if type(audit_convolution) is not bool or (audit_convolution and not self.fused_convolution):
            raise ValueError('Convolution audit requires the explicit fused candidate')
        stage('borrowed-embedding')
        local = retain(self.model.embd(identifiers, memory_config=operations.DRAM_MEMORY_CONFIG))
        local = retain(operations.reshape(local, (1, 1, self.block_rows, 2560)))
        stage('embedding-all-gather')
        hidden = retain(operations.experimental.all_gather_async(local, persistent_output_buffer=None, dim=3,
            multi_device_global_semaphore=self.collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.collectives.get_and_cycle_barrier_semaphore_handle(), num_links=projection_links(),
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2))
        if self.block_rows != 32:
            hidden = retain(operations.pad(hidden, [(0, 0), (0, 0), (0, 32 - self.block_rows), (0, 0)], 0.0))
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
                    retain, parameters=attention, context=context, **convolution_options,
                    **(dict(cached_history=cached_history[layer]) if cached_history is not None else {}))
            stage('mlp', layer=layer)
            hidden = execute_mlp_branch(operations, self.mesh, self.collectives, hidden, weights, convolution,
                retain, parameters=mlp, trace_safe=True, **convolution_options)['output']
        stage('final-norm-and-selector-projection')
        normalized = retain(operations.rms_norm(hidden, epsilon=1e-6, weight=self.final_norm,
            compute_kernel_config=self.kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 1),
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        projected = retain(operations.matmul(normalized, self.selector_projection, dtype=operations.float32,
            program_config=program, compute_kernel_config=self.kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
        projected = retain(operations.typecast(projected, operations.bfloat16))
        block = retain(operations.slice(normalized, (0, 0, 0, 0), (1, 1, self.block_rows, 5120)))
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
            raise AssertionError('Replicated learned selector features differ')
        selector_hidden = projected_parts[0][..., 1:self.block_rows, :].reshape(1, self.max_drafts, 256)
        tokens, unused = select_active_candidates(selector_hidden, candidates, unary, self.predecessors, self.successors,
            torch.tensor([seed], dtype=torch.int64))
        return tuple(int(token) for token in tokens[0, :count])

    def propose(self, seed, count):
        import torch

        if self.closed or self.pending is not None or type(seed) is not int or not 0 <= seed < 248320 or type(count) is not int or not 1 <= count <= self.max_drafts:
            raise ValueError('Committed DFlash2 history and bounded anchor/proposal IDs required')
        if self.proposal_capture is not None:
            tokens = self.proposal_capture.propose(seed, count)
            self.proposal_calls += 1
            return tokens
        operations = self.operations
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
            stage('upload-anchor-and-mask')
            identifiers = upload(torch.tensor([[seed, *([248070] * self.max_drafts)]], dtype=torch.int64), dtype=operations.uint32, row_major=True)
            stage('prepare-history-mask-and-rope')
            host_mask = draft_attention_mask(self.history_rows, block_rows=self.block_rows)
            key_rows = host_mask.shape[-1]
            valid_history = retain(operations.slice(self.history, (0, 0, 0, 0), (1, 1, self.history_rows, 5120)))
            history = retain(operations.pad(valid_history, [(0, 0), (0, 0), (0, key_rows - self.history_rows), (0, 0)], 0.0))
            mask = upload(host_mask)
            rope = {name: tuple(upload(value) for value in rope_tables(start, rows))
                for name, start, rows in (('q', self.position, 32), ('k', self.position - self.history_rows, key_rows))}
            outputs = self.execute_proposal(identifiers, history, mask, rope, context=self.history_rows,
                owned=owned, retain=retain, stage=stage)
            stage('synchronize-head-and-selector')
            operations.synchronize_device(self.mesh)
            stage('read-candidates-and-select')
            tokens = self.select_proposal(outputs, seed, count)
            self.proposal_calls += 1
            return tokens
        finally:
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
        if self.history is not None:
            self.operations.deallocate(self.history)
            self.history = None
        if self.spare_history is not None:
            self.operations.deallocate(self.spare_history)
            self.spare_history = None
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True
