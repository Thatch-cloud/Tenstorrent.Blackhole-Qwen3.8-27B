"""Explicit profiling-only adaptation of the admitted 32K shared-Q/K runtime."""

from frozen_recipe_context import replace_once


FILES = ('dspark_request_experiment.py', 'full_dspark_request.py',
    'request_verifier_profile_report.py', 'dspark-combined-profile.sh')


def adapt_sources(sources):
    result = dict(sources)
    changes = {
        'dspark_request_experiment.py': (
            ("'QWEN_GDN_OUTPUT_L1_EXPERIMENT', 'QWEN_GDN_OUTPUT_GRID_EXPERIMENT',\n"
                "                    'QWEN_COMBINED_TRACE_PROFILE'",
                "'QWEN_GDN_OUTPUT_L1_EXPERIMENT', 'QWEN_GDN_OUTPUT_GRID_EXPERIMENT'"),
            ("    if combined_profile:\n        schedule = (('publication', True),)",
                "    if combined_profile:\n"
                "        if os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1' or shared_flag != '1':\n"
                "            raise ValueError('Profile only the admitted shared-Q/K candidate')\n"
                "        schedule = (('publication', True),)")),
        'full_dspark_request.py': (
            ('gdn_outer_add or gdn_copy_pairs or gdn_output_l1 or gdn_output_grid or combined_profile or not (',
                'gdn_outer_add or gdn_copy_pairs or gdn_output_l1 or gdn_output_grid or not ('),),
        'request_verifier_profile_report.py': (
            ('from dspark_publication_variants import validate_route as publication_route',
                'from gdn_shared_qk_variants import validate_route as publication_route'),
            ("request.get('length') != 4096 or request.get('lookup_max_rows') != 16",
                "request.get('length') != 32768 or report.get('ctx_tokens') != 32768\n"
                "                or report.get('combined_runtime_profile') is not True\n"
                "                or request.get('lookup_max_rows') != 16")),
        'dspark-combined-profile.sh': (
            ('timeout -k 30 4200', 'timeout -k 30 720'),
            ('--max-new-tokens 64', '--max-new-tokens 256')),
    }
    for name, replacements in changes.items():
        for before, after in replacements:
            result[name] = replace_once(result[name], before, after)
        if name.endswith('.py'):
            compile(result[name], name, 'exec')
    return result
