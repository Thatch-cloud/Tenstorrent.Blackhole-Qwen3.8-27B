"""Static-position grouped reader for bounded real-weight attention experiments."""

from attention_batch import SerialAttentionReader
from attention_head_fold import causal_mask, chunk_groups, device_layout
from gdn_multitoken_conv import addresses, release_owned


class GroupedAttentionReader:
    def __init__(self, operations, mesh, start, rows, pages_host, positions, pages, upload):
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
        groups = chunk_groups(start, rows)
        if any(group['signature'][1] % 64 or group['signature'][1] // 64 > pages_host.shape[1] for group in groups):
            raise ValueError('Whole-page bounded group required')
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
        protected = {addresses(operations, value) for value in (query, keys, values)}
        try:
            for group, pages, mask, config in self.metadata:
                packed = device_layout(operations, query, group['rows'], owned, offset=group['offset'])
                options = dict(kwargs, program_config=config, is_causal=False, attn_mask=mask)
                result = operations.transformer.paged_scaled_dot_product_attention_decode(packed, keys, values,
                    page_table_tensor=pages, **options)
                owned.append(result)
                chunks.append(device_layout(operations, result, group['rows'], owned, inverse=True))
            output = operations.concat(chunks, dim=1, memory_config=kwargs['memory_config'])
            protected.add(addresses(operations, output))
            return output
        finally:
            release_owned(operations, [value for value in owned if addresses(operations, value) not in protected])

    def close(self):
        release_owned(self.operations, self.owned)
        self.owned.clear()
