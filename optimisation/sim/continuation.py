"""Opt-in single-owner text prefill continuation; requires explicit scheduler metadata."""

import math

import torch


class ContinuationController:
    def __init__(self, backend):
        self.backend = backend
        self.active = None
        self.poisoned = False

    def cancel(self, request_id, epoch):
        if self.active is None or self.active["owner"] != (request_id, epoch):
            raise ValueError("Cancellation does not own the prefill scratch")
        self.backend.drain()
        self.active = None
        self.poisoned = False

    def forward(self, tokens, page_table, prompt_lens, metadata):
        required = ("start_pos", "intermediate_prefill_mask", "request_ids", "request_epochs", "empty_slots")
        if self.poisoned:
            raise RuntimeError("Prefill scratch is poisoned; cancel its owner after a successful drain")
        if any(name not in metadata for name in required):
            raise ValueError(f"Continuation requires {required}")
        if tokens.ndim != 2 or tokens.shape[0] != 1 or page_table.ndim != 2 or page_table.shape[0] != 1:
            raise ValueError("Continuation v1 admits exactly one text prefill per lane")
        if prompt_lens is None or len(prompt_lens) != 1 or any(len(metadata[name]) != 1 for name in required):
            raise ValueError("Continuation metadata must contain exactly one row")
        if tokens.device.type != "cpu" or page_table.device.type != "cpu":
            raise ValueError("Continuation expects host token IDs and page table")
        if tokens.dtype not in (torch.int32, torch.int64) or page_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("Token IDs and page table must be integer tensors")
        for name in ("pixel_values", "pixel_values_videos"):
            value = metadata.get(name)
            absent_row = isinstance(value, list) and len(value) == 1 and value[0] is None
            if value is not None and not absent_row:
                raise ValueError("Continuation v1 is text-only")
        if metadata.get("vision_tokens") is not None:
            raise ValueError("Continuation v1 is text-only")
        def integer(value):
            if isinstance(value, torch.Tensor):
                if value.dtype not in (torch.int32, torch.int64) or value.numel() != 1:
                    raise ValueError("Positions and slots must be scalar integers")
                value = value.item()
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Positions and slots must be integers")
            return value

        start = integer(metadata["start_pos"][0])
        end = integer(prompt_lens[0])
        slot = integer(metadata["empty_slots"][0])
        intermediate = metadata["intermediate_prefill_mask"][0]
        if isinstance(intermediate, torch.Tensor):
            if intermediate.dtype != torch.bool or intermediate.numel() != 1:
                raise ValueError("Intermediate-prefill marker must be boolean")
            intermediate = intermediate.item()
        if not isinstance(intermediate, bool):
            raise ValueError("Intermediate-prefill marker must be boolean")
        final = not intermediate
        request_id = metadata["request_ids"][0]
        epoch = metadata["request_epochs"][0]
        if not isinstance(request_id, str) or not request_id or not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Request identity requires a nonempty ID and nonnegative integer epoch")
        owner = (request_id, epoch)
        chunk_size = self.backend.chunk_size
        if not 0 <= start < end <= min(tokens.shape[1], self.backend.max_length):
            raise ValueError("Invalid continuation token range")
        if start % chunk_size or end - start > chunk_size or (not final and end - start != chunk_size):
            raise ValueError("Non-final steps must be one aligned full chunk; only the final step may have a tail")
        if not 0 <= slot < self.backend.max_slots:
            raise ValueError("Invalid decode slot")
        used_pages = math.ceil(end / self.backend.block_size)
        if page_table.shape[1] < used_pages or used_pages > self.backend.page_capacity:
            raise ValueError("Insufficient page-table capacity")
        if torch.any(page_table[:, :used_pages] < 0) or torch.any(page_table[:, :used_pages] >= self.backend.physical_pages):
            raise ValueError("Physical page ID outside the allocated pool")
        if torch.any(tokens[:, :end] < 0) or torch.any(tokens[:, :end] >= self.backend.vocab_size):
            raise ValueError("Token ID outside the model vocabulary")
        if self.active is None:
            if start != 0:
                raise ValueError("Continuation has no live scratch owner")
        else:
            if owner != self.active["owner"] or start != self.active["end"]:
                raise ValueError("Scratch owner or expected continuation position mismatch")
            if not torch.equal(tokens[:, :start], self.active["tokens"]):
                raise ValueError("Previously processed token prefix changed")
            previous_pages = self.active["pages"]
            if not torch.equal(page_table[:, :previous_pages.shape[1]], previous_pages):
                raise ValueError("Previously populated KV page mapping changed")
        self.active = {"owner": owner, "end": start, "tokens": tokens[:, :start].clone(),
                       "pages": page_table[:, :math.ceil(start / self.backend.block_size)].clone()}
        try:
            output = self.backend.run(tokens[:, :end], page_table, start, end, final, slot)
        except BaseException:
            self.poisoned = True
            raise
        if final:
            self.active = None
        else:
            self.active.update(end=end, tokens=tokens[:, :end].clone(), pages=page_table[:, :used_pages].clone())
        return output


class TPPrefillBackend:
    def __init__(self, model):
        import ttnn

        self.ttnn = ttnn
        self.model = model
        self.restart_required = False
        self.chunk_size = model._chunked_chunk_size
        self.max_length = model.args.max_seq_len
        self.max_slots = model.args.max_batch_size
        self.vocab_size = model.args.vocab_size
        self.block_size = model._paged_kv_caches[0][0].shape[2]
        self.physical_pages = min(cache.shape[0] for pair in model._paged_kv_caches if pair is not None for cache in pair)
        if model.num_devices != 2 or self.max_slots <= 1 or model._chunked_trace_id is None:
            raise RuntimeError("Continuation requires a warmed two-device batched text model with a chunk trace")
        if not self.chunk_size or self.chunk_size % self.block_size:
            raise RuntimeError("Chunk size must align with paged KV blocks")
        self.page_capacity = int(model._chunk_full_page_table_buf.shape[-1])

    def drain(self):
        if self.restart_required or getattr(self.model, "_prefill_failed_dma_refs", None) is not None:
            raise RuntimeError("Continuation worker requires restart; scratch or decode slot may be unsafe")
        try:
            self.ttnn.synchronize_device(self.model.device)
        except BaseException:
            self.restart_required = True
            raise

    def run(self, tokens, page_table, start, end, final, slot):
        model, ttnn = self.model, self.ttnn
        self.drain()
        previous = model._bind_gdn_prefill_scratch()
        logits = None
        try:
            if start == 0:
                model._build_request_rope(tokens, None)
            logits = model._prefill_traced_chunked_tp(
                tokens, page_table, end, end // self.chunk_size, self.chunk_size,
                end % self.chunk_size, start_pos=start, is_last=final)
            self.drain()
            if final:
                composer = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
                host_logits = ttnn.to_torch(logits, mesh_composer=composer).reshape(-1, model.args.vocab_size)[:1].float().view(1, 1, -1).clone()
                layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
                recurrent = [ttnn.to_torch(layer.rec_state, mesh_composer=composer).clone() for layer in layers]
                convolution = [[ttnn.to_torch(state, mesh_composer=composer).clone() for state in layer.conv_states] for layer in layers]
            else:
                if logits is not None:
                    raise RuntimeError("Intermediate prefill unexpectedly produced logits")
                host_logits = torch.zeros(1, 1, model.args.vocab_size)
        finally:
            try:
                self.drain()
                if logits is not None:
                    ttnn.deallocate(logits)
            finally:
                model._unbind_gdn_prefill_scratch(previous)
        if final:
            try:
                model._write_gdn_slot(slot, recurrent, convolution)
                self.drain()
            except BaseException:
                self.restart_required = True
                raise
        return host_logits, torch.zeros(1, dtype=torch.long)


def dispatch_prefill(wrapper, tokens, page_table, prompt_lens, metadata):
    controller = getattr(wrapper, "_prefill_continuation_controller", None)
    if controller is None:
        controller = ContinuationController(TPPrefillBackend(wrapper.model[0]))
        wrapper._prefill_continuation_controller = controller
    return controller.forward(tokens, page_table, prompt_lens, metadata)
