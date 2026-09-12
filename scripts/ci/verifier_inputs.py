"""Host staging for fixed-shape verifier traces; page ownership remains unchanged."""


def validate_tokens(tokens, rows, start, vocab_size, capacity):
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) or len(tokens) != rows:
        raise ValueError('Ticket must match the captured verifier bucket')
    if type(start) is not int or start < 0 or start + rows > capacity:
        raise ValueError('Ticket positions exceed the captured page-table capacity')
    if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise ValueError('Token outside target vocabulary')


def host_inputs(tokens, start, rope_dim, theta):
    import torch

    if type(rope_dim) is not int or rope_dim < 2 or rope_dim % 2 or theta <= 0:
        raise ValueError('Positive even rotary dimension and base required')
    positions = torch.arange(start, start + len(tokens), dtype=torch.int32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    frequencies = torch.outer(positions.float(), inv_freq)
    embedding = torch.cat([frequencies, frequencies], dim=-1)
    cos = embedding.cos().reshape(1, len(tokens), 1, rope_dim).to(torch.bfloat16)
    sin = embedding.sin().reshape(1, len(tokens), 1, rope_dim).to(torch.bfloat16)
    return torch.tensor(tokens, dtype=torch.int32).reshape(len(tokens), 1), positions, cos, sin


def stage_inputs(fixture, tokens, start):
    operations, model = fixture.operations, fixture.model
    validate_tokens(tokens, fixture.rows, start, model.args.vocab_size, fixture.pages.shape[1] * 64)
    replay_reader = getattr(fixture, 'replay_reader', None)
    if replay_reader is not None:
        replay_reader.validate(start)
    if len(fixture.singleton_positions) != fixture.rows:
        raise ValueError('Every B1 attention position buffer must be retained')
    token_values, positions, cos, sin = host_inputs(tokens, start, model.args.rope_head_dim, model.args.rope_theta)
    values = [(fixture.tokens, token_values, operations.uint32, operations.ROW_MAJOR_LAYOUT),
              (fixture.positions, positions, operations.int32, operations.ROW_MAJOR_LAYOUT),
              (fixture.cos, cos, operations.bfloat16, operations.TILE_LAYOUT),
              (fixture.sin, sin, operations.bfloat16, operations.TILE_LAYOUT)]
    values.extend((destination, positions[index:index + 1], operations.int32, operations.ROW_MAJOR_LAYOUT)
                  for index, destination in enumerate(fixture.singleton_positions))
    if replay_reader is not None:
        import torch
        words = torch.zeros(8, dtype=torch.int32)
        words[0] = start
        values.append((replay_reader.positions, words, operations.int32, operations.ROW_MAJOR_LAYOUT))
    if any(tuple(destination.shape) != tuple(value.shape) or destination.dtype != dtype or destination.layout != layout
           for destination, value, dtype, layout in values):
        raise ValueError('Staged metadata must preserve every captured tensor signature')
    destinations = [destination for destination, value, dtype, layout in values]
    addresses = [tuple(part.buffer_address() for part in operations.get_device_tensors(value)) for value in destinations]
    if any(len(pair) != 2 for pair in addresses) or any(len({pair[chip] for pair in addresses}) != len(addresses) for chip in range(2)):
        raise ValueError('Two independent chip-local buffers per metadata input required')
    staged = [operations.from_torch(value, device=None, dtype=dtype, layout=layout,
              mesh_mapper=operations.ReplicateTensorToMesh(model.mesh_device)) for destination, value, dtype, layout in values]
    try:
        try:
            for source, destination in zip(staged, destinations, strict=True):
                operations.copy_host_to_device_tensor(source, destination)
        finally:
            operations.synchronize_device(model.mesh_device)
        if [tuple(part.buffer_address() for part in operations.get_device_tensors(value)) for value in destinations] != addresses:
            raise AssertionError('Input staging replaced a captured buffer')
    except BaseException:
        if replay_reader is not None:
            replay_reader.failed = True
        raise
    if replay_reader is not None:
        replay_reader.start = start
