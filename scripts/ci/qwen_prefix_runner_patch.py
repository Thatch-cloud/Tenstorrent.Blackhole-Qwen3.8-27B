"""Conversation prefix reuse on the TT general path (G1): the runner and worker AST stage.

The TT prefix-reuse design (revision 2, 2026-09-26), section 2.0.1 item 3. Two things the model
graft needs from the plugin, both inert unless QWEN_PREFIX_REUSE=1:

1. Row request ids. The model finds each prefill row's committed grant by request id
   (qwen_prefix_registry.PrefixRegistry.grant_for), and the model call carries none today. The ids
   exist where the runner builds the step (model_runner.py:1269, row_req_ids, the same order as the
   token rows and prefill_empty_slots), and submit_prefill builds the model kwargs (:1784-1797).
   TTModelInput is a frozen dataclass and the only carrier between the two, so:
   - model_input.py: TTModelInput gains a last field, prefill_request_ids (default None);
   - model_runner.py _prepare_model_inputs: sets it to the row ids on a prompt step when
     QWEN_PREFIX_REUSE=1, else None;
   - model_runner.py submit_prefill: passes it as the kwarg REQUEST_IDS_KWARG ("request_ids") when
     set. The model reads it from **kwargs (qwen36_vllm.prefill_forward takes **kwargs, IMG
     qwen36_vllm.py:203).
   Other constructors of TTModelInput (the lane-mode builders in input_batch.py, dataclasses.replace
   at model_runner.py:1609) leave or carry the default. With the variable unset the kwargs are the
   plugin's own, byte for byte.
2. The block-size assertion (F7). The uniproc executor calls update_block_size_for_backend after
   load_model (vLLM v1/executor/uniproc_executor.py:69), which can re-align a hybrid model's block
   size (platforms/interface.py:590-630,750). TTWorker.initialize_from_config, the first worker call
   that holds the final KVCacheConfig, refuses to start unless cache_config.block_size and every
   KV group's spec block size are 64: the model's page tables and chunk page arithmetic assume it.

Anchors: every file is sha256-pinned (the source.sha256 pattern of the C2 graft) and every edit is
an exact text anchor inside a named method that must occur once. model_runner.py and model_input.py
are the bf77cd63 blobs; worker.py is either the blob or the P8 image's copy, which
serving_plugin_patch.patch_worker derives from it (that module has not changed since the P8 base
be9e184e, so the image's bytes are deterministic; not read from a built image - UNVERIFIED).

    python3 -B qwen_prefix_runner_patch.py [--check] /opt/qwen-fast-plugin/src/vllm_tt_plugin
"""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

REQUEST_IDS_KWARG = 'request_ids'
PLUGIN_REVISION = 'bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c'
MODEL_RUNNER_SHA256 = 'eed4d0fbe0a41fcb18515ad72c0587a05ffa32e615032dd79771331226919530'
MODEL_INPUT_SHA256 = '8adf4bac4daba576deb27111757bedad69c1d669eb5c15d0ad2d128af2040b54'
WORKER_SHA256 = {
    'e9371e868fd3d9fa901dbe1780b443379c8da96252723405cd6183cfab7fc943': 'bf77cd63 blob',
    '05b99d88a060ede31a686029c3e20a2abd34087dd4670e5fbfb955247406d696': 'P8 image (serving_plugin_patch.patch_worker)',
}
HOOK_TAG = 'qwen_prefix_runner_patch'

# model_input.py :: TTModelInput -- the last field, then the new one.
FIELD_ANCHOR = (
    '    # Prefill only: rows whose forward writes KV state but must not emit a\n'
    '    # sampled token, because more prompt tokens remain after this chunk.\n'
    '    # ``None`` for decode.\n'
    '    intermediate_prefill_mask: torch.Tensor | None = None\n'
)
FIELD_NEW = FIELD_ANCHOR + (
    '\n'
    '    # Prefill only, and only under QWEN_PREFIX_REUSE=1 (qwen_prefix_runner_patch): the request\n'
    '    # id of each row, in row order. submit_prefill hands it to the model as ``request_ids``.\n'
    '    prefill_request_ids: list[str] | None = None\n'
)

# model_runner.py :: TTModelRunner._prepare_model_inputs -- compute the ids beside the slots ...
IDS_ANCHOR = (
    '        row_req_ids = [input_batch.req_ids[i] for i in req_indices]\n'
    '        prefill_empty_slots = None\n'
    '        slot_remap = None\n'
)
IDS_NEW = (
    '        row_req_ids = [input_batch.req_ids[i] for i in req_indices]\n'
    '        # Qwen prefix reuse (G1, qwen_prefix_runner_patch): the model finds each prefill\n'
    '        # row\'s committed grant by request id. Only under QWEN_PREFIX_REUSE=1.\n'
    '        import os as _qwen_prefix_os\n'
    '\n'
    '        prefill_request_ids = (\n'
    '            list(row_req_ids)\n'
    '            if is_prompt and _qwen_prefix_os.environ.get("QWEN_PREFIX_REUSE") == "1"\n'
    '            else None\n'
    '        )\n'
    '        prefill_empty_slots = None\n'
    '        slot_remap = None\n'
)
# ... and hand them to the TTModelInput it returns.
BUILD_ANCHOR = (
    '            prefill_empty_slots=prefill_empty_slots,\n'
    '            intermediate_prefill_mask=intermediate_prefill_mask,\n'
    '        )\n'
)
BUILD_NEW = (
    '            prefill_empty_slots=prefill_empty_slots,\n'
    '            intermediate_prefill_mask=intermediate_prefill_mask,\n'
    '            prefill_request_ids=prefill_request_ids,\n'
    '        )\n'
)

# model_runner.py :: TTModelRunner.submit_prefill -- the kwarg.
SUBMIT_ANCHOR = (
    '        if empty_slots is not None:\n'
    '            kwargs["empty_slots"] = list(empty_slots)\n'
)
SUBMIT_NEW = SUBMIT_ANCHOR + (
    '        if model_input.prefill_request_ids is not None:\n'
    '            # Qwen prefix reuse (G1, qwen_prefix_runner_patch): the row request ids.\n'
    '            kwargs["' + REQUEST_IDS_KWARG + '"] = list(model_input.prefill_request_ids)\n'
)

# worker.py :: TTWorker.initialize_from_config -- the block-size assertion (F7).
WORKER_ANCHOR = (
    '        single-process lane mode has only one rank.\n'
    '        """\n'
    '        self.model_runner.initialize_kv_cache(kv_cache_config)\n'
)
WORKER_NEW = (
    '        single-process lane mode has only one rank.\n'
    '        """\n'
    '        _qwen_prefix = os.environ.get("QWEN_PREFIX_REUSE") == "1"\n'
    '        if _qwen_prefix:\n'
    '            # Qwen prefix reuse (G1, qwen_prefix_runner_patch, design F7): the executor\n'
    '            # re-derives the block size after load_model; the model\'s 64-token page tables\n'
    '            # and 2048-token chunk arithmetic need exactly 64.\n'
    '            _qwen_prefix_sizes = sorted(\n'
    '                {group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups}\n'
    '            )\n'
    '            if self.cache_config.block_size != 64 or _qwen_prefix_sizes != [64]:\n'
    '                raise RuntimeError(\n'
    '                    "prefix reuse refused: cache_config.block_size=%s, KV spec block "\n'
    '                    "sizes %s; exactly 64 is required"\n'
    '                    % (self.cache_config.block_size, _qwen_prefix_sizes)\n'
    '                )\n'
    '        self.model_runner.initialize_kv_cache(kv_cache_config)\n'
    '        if _qwen_prefix:\n'
    '            # The bring-up marker: the device KV dtype the model allocated (bfloat8_b under\n'
    '            # QWEN_SDPA_BF8=1; vLLM\'s own spec says bf16 either way).\n'
    '            _qwen_prefix_kv = getattr(self.model_runner, "kv_caches", None)\n'
    '            for _ in range(4):\n'
    '                if isinstance(_qwen_prefix_kv, (list, tuple)) and _qwen_prefix_kv:\n'
    '                    _qwen_prefix_kv = _qwen_prefix_kv[0]\n'
    '            logger.info(\n'
    '                "[PINDIAG] prefix: worker block_size=%d kv_groups=%d kv_dtype=%s",\n'
    '                self.cache_config.block_size,\n'
    '                len(kv_cache_config.kv_cache_groups),\n'
    '                getattr(_qwen_prefix_kv, "dtype", "?"),\n'
    '            )\n'
)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def method_span(source, class_name, method_name):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError('expected one class %s, found %d' % (class_name, len(classes)))
    methods = [node for node in classes[0].body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name]
    if len(methods) != 1:
        raise ValueError('expected one %s.%s, found %d' % (class_name, method_name, len(methods)))
    node = methods[0]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
    return start, node.end_lineno


def class_span(source, class_name):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError('expected one class %s, found %d' % (class_name, len(classes)))
    node = classes[0]
    return node.lineno - 1, node.end_lineno


def replace_once(source, span, old, new, what):
    lines = source.splitlines(keepends=True)
    start, end = span
    region = ''.join(lines[start:end])
    if region.count(old) != 1:
        raise ValueError('%s: expected one anchor in lines %d-%d, found %d'
                         % (what, start + 1, end, region.count(old)))
    lines[start:end] = [region.replace(old, new, 1)]
    result = ''.join(lines)
    ast.parse(result)
    return result


def refuse_patched(source, name):
    if HOOK_TAG in source:
        raise ValueError('%s already carries the prefix-reuse runner patch' % name)


def patch_model_input(source):
    refuse_patched(source, 'model_input.py')
    tree = ast.parse(source)
    node = [item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'TTModelInput'][0]
    last = node.body[-1]
    if not (isinstance(last, ast.AnnAssign) and getattr(last.target, 'id', None) == 'intermediate_prefill_mask'):
        raise ValueError('TTModelInput no longer ends with intermediate_prefill_mask')
    return replace_once(source, class_span(source, 'TTModelInput'), FIELD_ANCHOR, FIELD_NEW,
                        'TTModelInput.prefill_request_ids')


def patch_model_runner(source):
    refuse_patched(source, 'model_runner.py')
    span = method_span(source, 'TTModelRunner', '_prepare_model_inputs')
    source = replace_once(source, span, IDS_ANCHOR, IDS_NEW, '_prepare_model_inputs ids')
    span = method_span(source, 'TTModelRunner', '_prepare_model_inputs')
    source = replace_once(source, span, BUILD_ANCHOR, BUILD_NEW, '_prepare_model_inputs TTModelInput')
    span = method_span(source, 'TTModelRunner', 'submit_prefill')
    return replace_once(source, span, SUBMIT_ANCHOR, SUBMIT_NEW, 'submit_prefill request_ids')


def patch_worker(source):
    refuse_patched(source, 'worker.py')
    span = method_span(source, 'TTWorker', 'initialize_from_config')
    return replace_once(source, span, WORKER_ANCHOR, WORKER_NEW, 'initialize_from_config block size')


FILES = (
    ('model_input.py', patch_model_input),
    ('model_runner.py', patch_model_runner),
    ('worker.py', patch_worker),
)


def pinned(name, digest):
    """The description of a pinned input, or None."""
    if name == 'model_input.py':
        return 'bf77cd63 blob' if digest == MODEL_INPUT_SHA256 else None
    if name == 'model_runner.py':
        return 'bf77cd63 blob' if digest == MODEL_RUNNER_SHA256 else None
    return WORKER_SHA256.get(digest)


def stage(package, check_only=False):
    """Patch the three files of <package> in place; refuses, before writing anything, any file
    whose sha256 is not pinned."""
    package = Path(package)
    patched = {}
    report = {}
    for name, operation in FILES:
        data = (package / name).read_bytes()
        digest = sha256_hex(data)
        origin = pinned(name, digest)
        if origin is None:
            raise ValueError('%s: sha256 %s is not a pinned plugin %s %s'
                             % (package / name, digest, PLUGIN_REVISION[:8], name))
        patched[name] = operation(data.decode('utf-8')).encode('utf-8')
        report[name] = (digest, origin, sha256_hex(patched[name]))
    if not check_only:
        for name, data in patched.items():
            (package / name).write_bytes(data)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('package', help='the vllm_tt_plugin package directory')
    parser.add_argument('--check', action='store_true', help='verify the pins and the patch; write nothing')
    options = parser.parse_args(argv)
    report = stage(options.package, check_only=options.check)
    for name, (before, origin, after) in sorted(report.items()):
        print('%s %s %s (%s) -> %s' % ('checked' if options.check else 'staged', name, before, origin, after))
    return 0


if __name__ == '__main__':
    sys.exit(main())
