"""Deterministic two-chip publication fixtures and independent host prefix reference."""


def shapes():
    compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
    return compact + [(16, 24, 128, 128)] + [(1, 16, 5120)] * 4 + [
        (8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact


def host_layer(pattern, layer):
    import torch
    if type(pattern) is not int or pattern not in (0, 1) or type(layer) is not int or not 0 <= layer < 48:
        raise ValueError('Declared pattern and layer required')
    generator = torch.Generator().manual_seed(389113 + pattern * 48 + layer)
    values = [torch.randn((shape[0] * 2, *shape[1:]), generator=generator).bfloat16() for shape in shapes()]
    for value in values[15:]:
        value.fill_(float('nan'))
    return values


def expected(values, prefix):
    import torch
    if type(prefix) is not int or not 0 <= prefix <= 16:
        raise ValueError('T16 accepted prefix required')
    if [tuple(value.shape) for value in values] != [(shape[0] * 2, *shape[1:]) for shape in shapes()]:
        raise ValueError('Complete two-chip entry/history/native/checkpoint fixture required')
    selected = values[:5] if prefix == 0 else [torch.cat([
        values[5][chip * 16 + prefix - 1:chip * 16 + prefix] for chip in range(2)])] + [
        value[:, prefix - 1:prefix] for value in values[6:10]]
    result = values[:10] + [value.clone() for value in values[10:15]] + selected
    for chip in range(2):
        result[10][chip * 8:chip * 8 + 1] = selected[0][chip:chip + 1]
    for slot in range(1, 5):
        result[10 + slot][:, :1] = selected[slot]
    return result
