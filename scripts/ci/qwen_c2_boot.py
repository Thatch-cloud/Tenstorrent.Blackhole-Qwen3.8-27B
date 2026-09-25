"""Boot the C2 serving contract (serving_c2_contract) in every python process of the image.

Installed into site-packages beside qwen_c2_serving.pth, whose one line imports this module,
so it runs at interpreter start - before PYTHONPATH could be relied on (the Thatch layer
replaces it) and before any of vLLM or the fast path is imported. A sitecustomize would
not do: the distribution's own /usr/lib/python3.10/sitecustomize.py comes first on sys.path.
Off unless QWEN_C2_SERVING=1. A failure in the vLLM API server exits it (status 78) rather
than serving outside the qualified configuration.
"""

import os

if os.environ.get('QWEN_C2_SERVING') == '1':
    import sys

    sys.path.insert(0, '/experiment-scripts/ci')
    try:
        import serving_c2_contract
    finally:
        sys.path.remove('/experiment-scripts/ci')
    try:
        serving_c2_contract.boot()
    except Exception:
        import traceback

        serving_c2_contract.log('boot failed:\n%s', traceback.format_exc())
        if serving_c2_contract.is_api_server(getattr(sys, 'orig_argv', None)):
            os._exit(78)
