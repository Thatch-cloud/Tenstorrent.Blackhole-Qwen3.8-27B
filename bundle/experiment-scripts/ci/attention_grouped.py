"""Static-position grouped reader for bounded real-weight attention experiments."""

from attention_batch import SerialAttentionReader
from attention_head_fold import causal_mask, chunk_groups, device_layout, parallel_groups
from gdn_multitoken_conv import addresses, release_owned


class GroupedAttentionReader:
    def __init__(self, operations, mesh, start, rows, pages_host, positions, pages, upload, *, dma_layout=False, parallel=False,
                 max_group_rows=4):
        if type(dma_layout) is not bool:
            raise ValueError('Explicit boolean DMA selection required')
        if type(parallel) is not bool or (parallel and not dma_layout):
            raise ValueError('Parallel groups require explicit boolean selection and DMA layout')
        if type(max_group_rows) is not int or max_group_rows not in (4, 8) or (max_group_rows == 8 and not parallel):
            raise ValueError('Eight-row reader requires the explicit parallel DMA path')
        self.parallel = parallel
        self.mesh, self.dma_layout = mesh, dma_layout
        self.operations = operations
        self.rows = rows
        self.calls = 0
        self.metadata = []
        self.owned = []
        self.native = SerialAttentionReader(operations, positions, [pages] * rows)
        self.positions = positions
        self.pages = pages
        grid = mesh.compute_with_storage_grid_size()
        if rows < 8:
            return
        groups = chunk_groups(start, rows, max_group_rows=max_group_rows)
        if any(group['signature'][1] % 64 or group['signature'][1] // 64 > pages_host.shape[1] for group in groups):
            raise ValueError('Whole-page bounded group required')
        if parallel:
            import torch
            for bundle in parallel_groups(start, rows, max_group_rows=max_group_rows):
                chunk, capacity = bundle[0]['signature']
                group_pages = upload(pages_host[:, :capacity // 64].repeat(len(bundle), 1).contiguous(), operations.int32)
                mask = upload(torch.cat([causal_mask(group['rows'], start + group['offset'], capacity)
                    for group in bundle], dim=0))
                self.owned.extend((group_pages, mask))
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=chunk)
                self.metadata.append((bundle, group_pages, mask, config))
            return
        for group in groups:
            chunk, capacity = group['signature']
            group_pages = upload(pages_host[:, :capacity // 64].contiguous(), operations.int32)
            mask = upload(causal_mask(group['rows'], start + group['offset'], capacity))
            self.owned.extend((group_pages, mask))
            config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                exp_approx_mode=False, q_chunk_size=0, k_chunk_size=chunk)
            self.metadata.append((group, group_pages, mask, config))

    def __call__(self, query, keys, values, *, page_table_tensor, cur_pos_tensor, **kwargs):
        operations = self.operations
        if tuple(query.shape) != (1, self.rows, 12, 256):
            raise ValueError('Static TP2 query geometry changed')
        self.calls += 1
        if self.rows == 1:
            return operations.transformer.paged_scaled_dot_product_attention_decode(query, keys, values,
                page_table_tensor=self.pages, cur_pos_tensor=self.positions[0], **kwargs)
        if self.rows < 8:
            return self.native(query, keys, values, page_table_tensor=page_table_tensor, cur_pos_tensor=cur_pos_tensor, **kwargs)
        owned, chunks = [], []
        def layout(tensor, count, *, inverse=False, offset=0):
            if self.dma_layout:
                from attention_fold_dma import device_layout_dma
                return device_layout_dma(self.mesh, tensor, count, owned, inverse=inverse, offset=offset)
            return device_layout(operations, tensor, count, owned, inverse=inverse, offset=offset)

        protected = {addresses(operations, value) for value in (query, keys, values)}
        try:
            if self.parallel:
                from attention_parallel import execute
                output = execute(self.mesh, operations, query, keys, values, self.metadata, owned,
                    scale=kwargs['scale'], memory_config=kwargs['memory_config'])
                protected.add(addresses(operations, output))
                return output
            for group, pages, mask, config in self.metadata:
                packed = layout(query, group['rows'], offset=group['offset'])
                options = dict(kwargs, program_config=config, is_causal=False, attn_mask=mask)
                result = operations.transformer.paged_scaled_dot_product_attention_decode(packed, keys, values,
                    page_table_tensor=pages, **options)
                owned.append(result)
                chunks.append(layout(result, group['rows'], inverse=True))
            output = operations.concat(chunks, dim=1, memory_config=kwargs['memory_config'])
            protected.add(addresses(operations, output))
            return output
        finally:
            release_owned(operations, [value for value in owned if addresses(operations, value) not in protected])

    def close(self):
        release_owned(self.operations, self.owned)
        self.owned.clear()
