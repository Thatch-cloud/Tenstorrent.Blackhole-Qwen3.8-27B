"""Source-preserving diagnostic adapter; no numerical or dataflow program changes."""

from gdn_multitoken import replace_once


def replacements():
    return (
        ('from gdn_shared_qk_program import build as build_normalization',
         'import hashlib\n'
         'from gdn_wait_clock import instrument\n'
         'from mlp_compute_clock_projection import validate_buffer\n'
         'from gdn_shared_qk_program import build as build_normalization\n\n'
         'BUILD_RECORDS = []'),
        ('def build_recurrence(ttnn, mesh, shards, kernels):',
         'def build_recurrence(ttnn, mesh, shards, kernels, token):'),
        ("    coordinates = core_coordinates(grid.x, grid.y, spec['workers'])",
         "    coordinates = core_coordinates(grid.x, grid.y, spec['workers'])\n"
         '    if coordinates[0] != (0, 0):\n'
         "        raise ValueError('Diagnostic sample owner must be logical core zero')"),
        ('                runtime[horizontal][vertical] = runtime_args(spec, role, worker, rows, addresses)',
         '                values = runtime_args(spec, role, worker, rows, addresses)\n'
         "                if role == 'compute':\n"
         '                    if values != [16]:\n'
         "                        raise ValueError('Unchanged recurrence token count required')\n"
         '                    values = values + [addresses[11], int((horizontal, vertical) == (0, 0)), token]\n'
         '                runtime[horizontal][vertical] = values'),
        ('def build(operations, mesh, tensors, *, root):',
         'def build(operations, mesh, tensors, *, root, samples, token=8):\n'
         '    if type(token) is not int or not 0 <= token < 16:\n'
         "        raise ValueError('Explicit T16 sample token required')\n"
         '    validate_buffer(operations, mesh, samples)'),
        ('    kernels = load_kernels(root)\n    recurrence = build_recurrence(operations, mesh, shards, kernels)',
         '    kernels = load_kernels(root)\n'
         "    before = kernels['recurrence']['compute']\n"
         '    after = instrument(before)\n'
         "    kernels['recurrence']['compute'] = after\n"
         '    BUILD_RECORDS.append(dict(control_sha256=hashlib.sha256(before.encode()).hexdigest(),\n'
         '        candidate_sha256=hashlib.sha256(after.encode()).hexdigest(), token=token))\n'
         '    recurrence = build_recurrence(operations, mesh,\n'
         '        shards + [operations.get_device_tensors(samples)], kernels, token)'),
        ('(tensors, recurrence), (tensors, norm_gate)',
         '(tensors + [samples], recurrence), (tensors, norm_gate)'),
    )


def instrument_pipeline(source):
    result = source
    for before, after in replacements():
        result = replace_once(result, before, after)
    compile(result, 'gdn_wait_clock_pipeline.py', 'exec')
    if remove_pipeline(result) != source:
        raise ValueError('Pipeline diagnostic cannot be removed exactly')
    return result


def remove_pipeline(source):
    for before, after in reversed(replacements()):
        source = replace_once(source, after, before)
    return source
