"""Two-tile decode norm configs for a verify block wider than one 32-row tile.

WHY. Run 35502452429 (image v48, four T16 users in one 64-row packed block) died at
attach, in the block's warm forward, before any block adapter ran:

    model.py:938 _forward_decode -> layer.py:188 attention_norm
    -> distributed_norm.py:85 all_gather_async
    TT_FATAL tensor_spec.cpp:161: Shard height 32 must match physical height 64 for width sharded

The model's decode mode builds its norm activation specs one tile high. Every decode
norm on the block's path takes its config from ONE getter, `args.get_norm_config`:
layer.py:171 and :177 (`("attn", Mode.DECODE)` for both the attention norm and, by the
model's own choice, the ff norm) and model.py:521 (`("lm_head", Mode.DECODE)`, the final
norm). The getter lives in the framework ModelArgs (tt_transformers), not in the qwen36
model_config.py, and its dict is read at exactly two places on the decode path:

- distributed_norm.py:57, :81, :94: `sharded_output_config`, the memory config the
  pre-norm all-gather writes its output in - a WIDTH-sharded spec whose shard is
  (32, dim / cores). That is the spec the 64-row tensor was refused against.
- rmsnorm.py:138-139, :145-146: `sharded_program_config` (a
  LayerNormShardedMultiCoreProgramConfig with block_h 1, one tile row per core) and the
  same `sharded_output_config`, handed to ttnn.rms_norm as program_config and
  memory_config when the input is sharded (in_sharded=True, out_sharded=True, which is
  what DistributedNorm.forward passes in decode, distributed_norm.py:108-110).

WHAT. `two_tile_norm_config` rebuilds those two values for the block's rows: the same
core grid, orientation and buffer type with a (rows, width) shard, and the same grid,
subblock_w and block_w with block_h = rows / 32. Every other key passes through
untouched (whatever else the framework carries), except that any unknown value carrying
a one-tile shard is refused, because leaving it on the path would fail the way the
gather did, and that a None `output_mem_config` becomes L1-interleaved: rmsnorm.py:176-177
then hands the norm output on interleaved, which is the layout the consumers of a
wide block take (two_tile_decode.py: the GDN in-projection's 2D branch and the MLP's
unfused prefill arm read the activation as given, and the 1D attention arm's own
interleave becomes a no-op). The framework dict carries None there for the layer norms
(tt_transformers model_config.py:2196); model.py:522 sets DRAM for the final norm after
the getter returns, and that assignment wins as before. `bind_two_tile_norms` wraps the
getter for one block: decode configs come back two tiles high, prefill and every other
call reach the native getter unchanged, and the wrapper counts its calls so
ModelBatch.run can check that every decode norm of the forward (two per layer plus the
final norm) took the rebuilt config.

WHERE IT RUNS. ModelBatch adds the binding only for a block wider than one tile
(rows > 32) and applies it, like every adapter, only around that block's forward through
instance_overrides. The 32-row block builds no binding: its calls are byte for byte the
ones the two-user M1 gate passed on image v47.

COST. The pre-norm gather moves (rows, dim / tp) BF16 per chip: at 64 rows 320 KB instead
of 160 KB, about 2 us more per gather on the 84 GB/s link, 129 gathers per forward -
roughly 0.3 ms per verify over a per-collective fixed cost that does not change. The
sharded norm at block_h 2 does twice the one-tile work on the same cores, a few
microseconds per norm. Neither adds a weight pass.

UNVERIFIED ON HARDWARE until the four-user gate runs: ttnn.rms_norm's sharded program
at block_h 2 over a width-sharded 64-row input (rmsnorm.py:9 carries a framework note
that the sharded rms_norm once required a single-tile shard height), the 64-row
all_gather_async into a two-tile width-sharded output, and the layout the rebuilt norm
output presents to the model's attention and GDN input projections at 64 rows.
"""

TILE = 32
SHARDED_KEYS = ('sharded_output_config', 'sharded_program_config')


def validate_two_tile_rows(rows):
    """The block's rows as whole tiles beyond one: 64 is the M3 block. One tile or less
    never takes this path (the model's own one-tile configs serve it)."""
    if type(rows) is not int or rows <= TILE or rows % TILE:
        raise ValueError('Two-tile norm configs serve blocks of whole %d-row tiles beyond one tile; %r rows given'
                         % (TILE, rows))
    return rows // TILE


def is_decode(mode):
    """The model passes tt_transformers Mode.DECODE (layer.py:168, model.py:521)."""
    return getattr(mode, 'name', None) == 'DECODE' or mode == 'decode'


def shard_height(value):
    """The shard height of a sharded memory config, or None for anything else."""
    spec = getattr(value, 'shard_spec', None)
    shape = getattr(spec, 'shape', None) if spec is not None else None
    if shape is None:
        return None
    try:
        return int(tuple(shape)[0])
    except (TypeError, ValueError, IndexError):
        return None


def two_tile_memory_config(config, rows, operations):
    """The model's one-tile WIDTH-sharded activation spec, rebuilt `rows` high on the same
    cores, in the same orientation and buffer type."""
    validate_two_tile_rows(rows)
    spec = getattr(config, 'shard_spec', None)
    layout = getattr(config, 'memory_layout', None)
    if spec is None or layout != operations.TensorMemoryLayout.WIDTH_SHARDED:
        raise ValueError('The decode norm output config must be a width-sharded activation spec; the model changed')
    height, width = (int(size) for size in tuple(spec.shape))
    if height != TILE:
        raise ValueError('The decode norm output config is expected one tile (%d rows) high, found %d: the model changed'
                         % (TILE, height))
    return operations.MemoryConfig(layout, config.buffer_type,
                                   operations.ShardSpec(spec.grid, [rows, width], spec.orientation))


def two_tile_program_config(config, rows, operations):
    """The model's one-tile sharded norm program, rebuilt with block_h = rows / 32 on the
    same grid with the same subblock_w, block_w and inplace flag."""
    tiles = validate_two_tile_rows(rows)
    names = ('compute_with_storage_grid_size', 'subblock_w', 'block_h', 'block_w', 'inplace')
    missing = [name for name in names if not hasattr(config, name)]
    if missing:
        raise ValueError('The decode norm program config does not expose %s; it cannot be rebuilt two tiles high'
                         % ', '.join(missing))
    if config.block_h != 1:
        raise ValueError('The decode norm program config is expected one tile row per core (block_h 1), found %r: '
                         'the model changed' % (config.block_h,))
    grid = config.compute_with_storage_grid_size
    return operations.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), subblock_w=config.subblock_w, block_h=tiles,
        block_w=config.block_w, inplace=config.inplace)


def two_tile_norm_config(config, rows, operations, output_mem_config=None):
    """The framework's decode norm config dict with its two sharded values rebuilt for
    `rows`; every other key passes through, except that a None `output_mem_config` takes
    `output_mem_config` when one is given (the wide block's interleaved hand-off). Refuses a
    dict without the sharded keys (the consumers above would then run the norm interleaved,
    which is not the 32-row path this mirrors) and any other value that still carries a
    one-tile shard."""
    validate_two_tile_rows(rows)
    if not isinstance(config, dict) or any(key not in config for key in SHARDED_KEYS):
        raise ValueError('The decode norm config no longer carries %s; the two-tile override cannot rebuild it'
                         % ' and '.join(SHARDED_KEYS))
    rebuilt = dict(config)
    rebuilt['sharded_output_config'] = two_tile_memory_config(config['sharded_output_config'], rows, operations)
    rebuilt['sharded_program_config'] = two_tile_program_config(config['sharded_program_config'], rows, operations)
    stale = sorted(key for key, value in rebuilt.items() if key not in SHARDED_KEYS and shard_height(value) == TILE)
    if stale:
        raise ValueError('Decode norm config key(s) %s carry a one-tile shard the override does not rebuild'
                         % ', '.join(stale))
    if output_mem_config is not None and rebuilt.get('output_mem_config') is None:
        rebuilt['output_mem_config'] = output_mem_config
    return rebuilt


class TwoTileNormBinding:
    """The one instance binding for a wide block: `args.get_norm_config` answering decode
    calls two tiles high, their output interleaved in L1. `binding` is the (instance, name,
    value) triple ModelBatch applies through instance_overrides (`bindings` lists it, the
    shape every two-tile binder shares); `calls` counts the decode configs rebuilt."""

    label = 'decode norm'

    def __init__(self, args, rows, operations, expected_calls):
        self.rows = rows
        self.tiles = validate_two_tile_rows(rows)
        self.args, self.operations = args, operations
        self.expected_calls = expected_calls
        native = getattr(args, 'get_norm_config', None)
        if not callable(native):
            raise ValueError('The model args expose no get_norm_config to rebuild; the model changed')
        if getattr(operations, 'L1_MEMORY_CONFIG', None) is None:
            raise ValueError('An interleaved L1 memory config is required for the wide block norm output')
        self.native = native
        self.output_mem_config = operations.L1_MEMORY_CONFIG
        self.calls = 0
        self.binding = (args, 'get_norm_config', self)
        self.bindings = [self.binding]

    def __call__(self, name, mode, *rest, **options):
        config = self.native(name, mode, *rest, **options)
        if not is_decode(mode):
            return config
        self.calls += 1
        return two_tile_norm_config(config, self.rows, self.operations, output_mem_config=self.output_mem_config)


DECODE_NORMS = ('attn', 'lm_head')  # layer.py:171 and :177 (both "attn"), model.py:521


def decode_mode():
    """The model's Mode.DECODE (tt_transformers common.py), the value layer.py:168 and
    model.py:521 pass the getter."""
    from models.tt_transformers.tt.common import Mode

    return Mode.DECODE


def bind_two_tile_norms(model, rows, operations, mode=None):
    """The wide block's norm binding over the model's args: two decode norms per layer and
    the final norm (model.py:521) are the calls one forward makes.

    Both decode configs the forward will ask for are rebuilt here, at attach, before any
    device op: a config the rebuild refuses (a shard that is not WIDTH, a program that is
    not one tile) fails the attach by name instead of raising on the host at the end of a
    forward whose ops are already enqueued (the final norm's config is the LAST call of the
    forward, model.py:521). The getter is pure, so the trial rebuild changes nothing and
    counts nothing."""
    layers = getattr(model, 'layers', None)
    if layers is None or getattr(model, 'args', None) is None:
        raise ValueError('A model with layers and args is required')
    binding = TwoTileNormBinding(model.args, rows, operations, expected_calls=2 * len(layers) + 1)
    mode = decode_mode() if mode is None else mode
    for name in DECODE_NORMS:
        try:
            two_tile_norm_config(binding.native(name, mode), rows, operations, output_mem_config=binding.output_mem_config)
        except ValueError as failure:
            raise ValueError('The %r decode norm config cannot be rebuilt for %d rows: %s' % (name, rows, failure)) from failure
    return binding
