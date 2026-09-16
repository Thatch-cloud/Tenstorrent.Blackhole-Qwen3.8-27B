"""Prepare historical target replay for selected geometry and runtime BF16 KV."""

from frozen_recipe_context import replace_once


def adapt_target_probe(source):
    source = replace_once(source, 'from pathlib import Path',
        'from pathlib import Path\nfrom frozen_context_geometry import selected_geometry')
    source = replace_once(source,
        "scope='T16 long-context attention component, at CTX8192; not full-model correctness or TG',",
        "scope='Selected-context BF16 KV target replay; not full-model correctness or TG',\n"
        "        context=selected_geometry()['context'], kv_dtype='bfloat16',")
    source = replace_once(source, "'target-t16-attention-8k-probe.py')}",
        "'target-t16-attention-8k-probe.py', 'frozen_context_geometry.py')}")
    source = replace_once(source, 'for capacity in (8448,):',
        "for capacity in (selected_geometry()['capacity'],):")
    for name in ('keys', 'values'):
        source = replace_once(source,
            f'{name} = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)',
            f'{name} = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat16)')
    compile(source, 'target-t16-attention-8k-probe.py', 'exec')
    return source
