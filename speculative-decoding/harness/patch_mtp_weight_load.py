# SPDX-License-Identifier: Apache-2.0
"""K3 — make MTP weight loading opt-in in tt-metal's qwen36 `weight_mapping.py`.

Answers #442 Q7 ("is MTP weight load opt-in?") by construction.

`weight_mapping.py` drops every `mtp.*` key at two sites, and `test_no_mtp_keys` asserts it
stays that way. This gates both behind `QWEN36_LOAD_MTP=1`, default off, so behaviour is
byte-identical unless the flag is set and the existing guard test still passes.

Deliberately minimal: MTP keys already fall through to the general "any remaining keys pass
through" branch (they do not start with `layers.`, and no rename rule matches), so they arrive
under their original `mtp.*` names with no extra mapping code. Keeping the divergence to two
conditions and an import matters if this becomes an upstream PR.

Applied by mounting the patched file into the tt-serving image. Idempotent, and it verifies its
anchors so an upstream edit fails loudly rather than silently patching nothing.

    python3 tt-patch-mtp-weight-load.py <path-to-weight_mapping.py>
"""
import io
import sys

ANCHORS = [
    (
        # site A -- remap_qwen36_state_dict
        '        # Filter out MTP (multi-token prediction) weights (original key)\n'
        '        if key.startswith("mtp"):\n'
        '            continue\n',
        '        # Filter out MTP (multi-token prediction) weights (original key) unless opted in.\n'
        '        if key.startswith("mtp") and not _load_mtp():\n'
        '            continue\n',
    ),
    (
        # site B -- load_qwen36_state_dict_fp8
        '        if "visual" in key or key.startswith("mtp"):\n'
        '            continue\n',
        '        if "visual" in key or (key.startswith("mtp") and not _load_mtp()):\n'
        '            continue\n',
    ),
    (
        'import json\nfrom pathlib import Path\n',
        'import json\nimport os\nfrom pathlib import Path\n',
    ),
    (
        '- Filtering out vision encoder and MTP weights\n',
        '- Filtering out vision encoder weights, and MTP weights unless QWEN36_LOAD_MTP=1\n',
    ),
]

HELPER = '''

def _load_mtp() -> bool:
    """True when MTP (multi-token prediction) weights should be kept.

    Off by default: the model does not use them and they cost ~0.8 GiB. With
    ``QWEN36_LOAD_MTP=1`` they pass through under their original ``mtp.*`` names -- deliberately
    NOT remapped, because ``mtp.layers.0.X`` collides with ``layers.N.X`` under any
    ``endswith``-style lookup (``load_layer_weights`` matches exactly that way, and for the MLP
    would otherwise return the *target's* layer 0).

    Read per call rather than at import so a test can toggle it without reimporting.
    """
    return os.environ.get("QWEN36_LOAD_MTP") == "1"
'''


def main(path):
    src = io.open(path, encoding="utf-8").read()
    if "_load_mtp" in src:
        print(f"PATCH already applied: {path}")
        return 0
    for old, new in ANCHORS:
        if old not in src:
            print(f"PATCH FAILED: anchor not found:\n{old!r}", file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)
    # insert the helper after the constants block, before the first def
    i = src.index("\ndef remap_qwen36_state_dict")
    src = src[:i] + HELPER + src[i:]
    io.open(path, "w", encoding="utf-8", newline="\n").write(src)
    print(f"PATCH applied: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
