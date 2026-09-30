"""Explicit candidate admission changes; never applied to default serving files."""

from frozen_recipe_context import replace_once, COMBINED_RUNTIME_CONTEXTS
from frozen_context_geometry import geometry


FILES = ('dspark_8k_admission.py', 'dspark_8k_entry.py', 'target_t16_attention_gate.py',
    'dspark_8k_build.py', 'dspark_runtime_cache.py', 'coding_context_request.py')


def adapt_combined_sources(sources, context=32768):
    # 32768 is the retained, evidence-qualified candidate (frozen_combined_gate.REPORTS);
    # its literals below must stay byte-identical to the pre-geometry-derived values.
    # 65536 stages the same shape from selected_geometry(); frozen_combined_gate.qualify()
    # still refuses it until its own evidence exists. The bare "8192"/"4096" literals left
    # untouched below are the unrelated 4K/8K qualification pilots (coding_context_request.py,
    # dspark_8k_admission.py's no-admission history_limit() fallback) or the T16 gate's
    # unrelated 4096-request base clause (target_t16_attention_gate.py) - none of those name
    # the combined-runtime candidate context and must not move with it.
    if context not in COMBINED_RUNTIME_CONTEXTS:
        raise ValueError('Only 32768 or 65536 are staged combined-runtime contexts')
    shape = geometry(context)
    request_context, capacity = shape['context'], shape['capacity']
    result = dict(sources)
    result['dspark_8k_admission.py'] = adapt_admission(result['dspark_8k_admission.py'], context=context)
    replacements = {
        'coding_context_request.py': (
            ('context_tokens not in (4096, 8192)', f'context_tokens not in (4096, 8192, {request_context})'),
            ('Only the explicit 4K and 8K context qualification pilots are enabled',
                f'Only explicit 4K, 8K and candidate {request_context} context pilots are enabled'),
            ('    sources = {name: Path(__file__).with_name(name).read_bytes() for name in filenames}',
                f"    if context_tokens == {request_context}:\n"
                "        filenames = (*EXTENDED_CONTEXT_FILES, 'full-prefix.py', 'full_dflash_request.py',\n"
                "            'full_dspark_request.py', 'dspark_request_experiment.py', 'gdn-prefix.py', 'learned-attention-probe.py')\n"
                '    sources = {name: Path(__file__).with_name(name).read_bytes() for name in filenames}')),
        'dspark-target-hardware.py': (
            ('if options.request and not options.preflight and request_context() == 8192:',
                f'if options.request and not options.preflight and request_context() == {request_context}:'),
            ("if history_limit() != 8448 or gate['native_reference'].get(SOURCE) != SOURCE_SHA256:",
                f"if history_limit() != {capacity} or gate['native_reference'].get(SOURCE) != SOURCE_SHA256:"),
            ("        gate['native_reference'] = dict(gate['native_reference'], **{SOURCE: qualified_factory})",
                "        gate['native_reference'] = dict(gate['native_reference'], **{SOURCE: qualified_factory})\n"
                '        from frozen_combined_runtime import qualified_native_reference\n'
                "        gate['native_reference'] = qualified_native_reference(root, gate['native_reference'])\n"
                "        native = native_fingerprints(root, dict(native_sources=gate['native_reference']))"),
            ('if options.preflight and request_context() == 8192:',
                f'if options.preflight and request_context() == {request_context}:'),
            ('from dspark_attention_8k_gate import qualify as qualify_8k',
                'from frozen_combined_runtime import qualify as qualify_8k')),
        'dspark_runtime_cache.py': (
            ('enabled=request_context() == 8192', f'enabled=request_context() == {request_context}'),),
        'dspark_8k_build.py': (
            ('from dspark_attention_8k_gate import qualify',
                'from frozen_combined_runtime import qualify, prepare_scratch, verify_scratch'),
            ('    admission = qualify(scripts)',
                '    admission = qualify(scripts)\n    scratch = prepare_scratch(root)'),
            ('key_chunk_size=256, capacity=8448)',
                f'key_chunk_size=256, capacity={capacity}, target_tree_scratch=scratch)'),
            ('    root = Path(root)\n',
                "    root = Path(root)\n    verify_scratch(root, inputs.get('target_tree_scratch'))\n")),
        'run-dspark-hardware.sh': (
            ('    -e "QWEN_DSPARK_MODE=$mode"',
                '    -e "QWEN_FROZEN_COMBINED_RUNTIME=${QWEN_FROZEN_COMBINED_RUNTIME:-0}" \\\n'
                '    -e "QWEN_SDPA_TREE_SCRATCH_ROUNDS=${QWEN_FROZEN_COMBINED_RUNTIME:-0}" \\\n'
                '    -e "QWEN_DSPARK_MODE=$mode"'),),
        'target_t16_attention_gate.py': (
            ('if request_context() == 8192:\n        from target_t16_attention_8k_gate import validate_request_option as validate_8k',
                f'if request_context() == {request_context}:\n        from frozen_combined_runtime import validate_target_option as validate_8k'),
            ('if request_context() == 8192:\n        from target_t16_attention_8k_gate import qualify as qualify_8k',
                f'if request_context() == {request_context}:\n        from frozen_combined_runtime import qualify_target as qualify_8k')),
        'dspark_8k_entry.py': (
            ('request_context() != 8192', f'request_context() != {request_context}'),
            ("    from dspark_8k_scope import runtime_scope",
                "    if os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1' or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':\n"
                "        raise ValueError('Explicit combined candidate and compact scratch required')\n"
                '    from dspark_8k_scope import runtime_scope'),
            ('context=8192, output_tokens=256', f'context={request_context}, output_tokens=256')),
        'dspark_8k_scope.py': (
            ('            FullHistoryKV.__init__(self, operations, mesh, collectives, parameters,',
                '            from frozen_combined_history import initialise_history\n'
                '            initialise_history(self, operations, mesh, collectives, parameters,'),
            ("        stack.enter_context(patch.object(dspark_full_attention, 'MAX_CONTEXT', geometry(context)['capacity']))",
                '        import dspark_prefill\n'
                '        import full_dspark_request\n'
                '        from frozen_combined_history import prefill_capture_class\n'
                '        capture = prefill_capture_class(dspark_prefill.FullHistoryCapture)\n'
                "        stack.enter_context(patch.object(dspark_prefill, 'FullHistoryCapture', capture))\n"
                "        stack.enter_context(patch.object(full_dspark_request, 'FullHistoryCapture', capture))\n"
                "        stack.enter_context(patch.object(dspark_full_attention, 'MAX_CONTEXT', geometry(context)['capacity']))"),
            ('from dspark_attention_8k_gate import REPORT_SHA256',
                'from frozen_combined_runtime import REPORT_SHA256'),
            ('        stack.enter_context(scoped_stats_pack())',
                '        from dspark_ladder_scalar_reciprocal import scalar_reciprocal\n'
                '        stack.enter_context(scalar_reciprocal())\n'
                '        from frozen_reciprocal_isolation import isolated_reciprocal\n'
                '        stack.enter_context(isolated_reciprocal())\n'
                '        stack.enter_context(scoped_stats_pack())')),
        'dspark_context_selection.py': (
            ('if history_limit() == 8448:', f'if history_limit() == {capacity}:'),),
    }
    for name, changes in replacements.items():
        for before, after in changes:
            result[name] = replace_once(result[name], before, after)
    return result


def adapt_admission(source, context=32768):
    if context not in COMBINED_RUNTIME_CONTEXTS:
        raise ValueError('Only 32768 or 65536 are staged combined-runtime contexts')
    shape = geometry(context)
    request_context, capacity = shape['context'], shape['capacity']
    changes = (
        ('from dspark_attention_8k_gate import qualify', 'from frozen_combined_runtime import qualify, verify_scratch'),
        ('    factory_sha256 = verify_factory(factory_root)',
            "    verify_scratch(factory_root, build_evidence.get('factory_inputs', {}).get('target_tree_scratch'))\n"
            '    factory_sha256 = verify_factory(factory_root)'),
        ('return 8448 if _ADMISSION.get() is not None else 8192',
            "return _ADMISSION.get()['capacity'] if _ADMISSION.get() is not None else 8192"),
        ('context != 8192 or output_tokens != 256', f'context != {request_context} or output_tokens != 256'),
        ('8K trial requires exactly 8192 prompt rows and 256 output-token headroom',
            f'Qualified candidate requires exactly {request_context} prompt rows and 256 output-token headroom'),
        ('output_tokens=output_tokens, capacity=8448, key_chunk_size=256',
            f'output_tokens=output_tokens, capacity={capacity}, key_chunk_size=256'),
    )
    for before, after in changes:
        source = replace_once(source, before, after)
    compile(source, 'dspark_8k_admission.py', 'exec')
    return source
