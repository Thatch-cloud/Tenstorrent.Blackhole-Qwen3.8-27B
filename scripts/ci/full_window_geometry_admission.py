"""Admit only the explicit full-window context-list addition to historical evidence."""

import hashlib


ORIGINAL = '03b9627aa16ca8ed53c7623b336ca462ceb5176f3b2df15ca15bc7cb70ea3672'
EXTENDED = 'a5bcc57042153ad09ff0dcc151bda16532b974d57c6afdc2d49574b746234c0f'


def staged_geometry(payload, *, full_window):
    if type(full_window) is not bool:
        raise ValueError('Explicit geometry source selection required')
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum == ORIGINAL and not full_window:
        return payload
    admit(payload, ORIGINAL, 261888)
    return payload if full_window else payload.replace(b', 261888', b'')


def admit(payload, expected, context):
    if (context != 261888 or type(context) is not int or expected != ORIGINAL
            or hashlib.sha256(payload).hexdigest() != EXTENDED
            or payload.count(b', 261888') != 1
            or hashlib.sha256(payload.replace(b', 261888', b'')).hexdigest() != ORIGINAL):
        raise ValueError('Only the pinned full-window context-list extension is admitted')
    return dict(original_sha256=ORIGINAL, extended_sha256=EXTENDED,
        change='Add 261888 to CONTEXTS only', requested_context=context,
        performance_qualified=False)


def adapt(source):
    from frozen_recipe_context import replace_once

    original = """        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Combined runtime component source differs: ' + name)"""
    replacement = """        payload = (directory / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum:
            if name != 'frozen_context_geometry.py':
                raise ValueError('Combined runtime component source differs: ' + name)
            from full_window_geometry_admission import admit
            result['geometry_source_extension'] = admit(payload, checksum, selected_geometry()['context'])"""
    result = replace_once(source, original, replacement)
    compile(result, 'frozen_combined_runtime.py', 'exec')
    return result
