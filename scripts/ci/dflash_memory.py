"""Single-request experimental KV allocation without reducing logical context."""


def request_kv_allocation(dflash_drafts):
    if dflash_drafts not in ('0', '7', '31'):
        raise ValueError('Explicit disabled or seven/31-proposal DFlash2 policy required')
    return dict(physical_pages=8200 if dflash_drafts == '0' else 1032, request_pages=1024,
        block_tokens=64, request_capacity_tokens=65536, gdn_slots=8,
        scope='Existing allocation' if dflash_drafts == '0' else
            'Single-request DFlash2 experiment: 1024 addressable KV pages plus eight spare pages; serving unchanged')
