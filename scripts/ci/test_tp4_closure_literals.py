"""No pair-only literal on the four-card serving path, except where a _tp twin replaces the function that carries it.

The S2 fast path was written for the audited p150a pair, so the modules it imports carry the pair's numbers as literals:
a (1, 4, N, 128) K/V bank (four KV heads per chip), the 24 GDN value heads, an 8,240-wide projected row, a 124,160-column
vocabulary shard, "both chips" checks. The four-card port cannot edit the 42+ sources recorded evidence pins (test_tp2_pins),
so it writes sibling `_tp` twins (widths from tp_shapes) and points the pair's names at them through the tp_addresses seam
(TWINS) at QWEN_FAST_TP=4 only. Two reviews of that port each found pair literals no test reached (publication_warm's
(1, 4, 2048, 128) bank and 2,048-wide query, dflash_proposal_trace's four KV heads): they lived in modules the attach
simulation fakes, and nothing looked at the code itself.

This is that look. It computes the serving import closure - the modules the c2-packed-tp4 attach and the first packed
round import, statically, from the roots below - scans every one of them for the pair's literals, and fails on any hit
that is not accounted for:

  - a hit inside a function (or class) the tp_addresses seam replaces is fine: that is what the twin is for. The twin
    itself is scanned too and must be clean;
  - a hit in a module the four-card profiles never run (flag-off arms, measurement harnesses) is fine only because the
    module is not in the closure: the closure stops at NOT_SERVED, each entry naming the flag or reason, and every
    LAZY (function-level) import of a module in the closure must be classified SERVED or NOT_SERVED, so a new lazy
    import of an unclassified module fails here instead of slipping past;
  - anything else needs an ALLOWED entry with the reason it cannot fail at four cards, and an entry that no longer
    matches a hit is itself a failure (so the list cannot rot).

A NEW pair literal on the four-card path - in a served module, or a new served module carrying one - therefore fails CI.
Source text only: nothing here imports a served module or touches a card."""

import ast
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# What the c2-packed-tp4 attach (serving_startup.start -> serving_runtime.attach_combined_runtime) and the first packed round
# (the worker hook's packed step: verify, publication, proposal) import at module level; everything else they reach lazily
# is classified below.
ROOTS = ('serving_startup', 'serving_runtime', 'serving_worker_hook', 'serving_packed_step', 'serving_sequential_step',
         'serving_buffer_pool', 'packed_verifier', 'publication_warm', 'dflash_proposal_trace', 'dflash_device',
         'dflash_combined_request', 'dflash_packed_proposal', 'dflash_packed_proposal_coordinator', 'model_batch',
         'packed_any_admission', 'serving_request_factory', 'serving_lifecycle', 'serving_c2_contract')

# Function-level imports the four-card profiles do run (an attach stage, a packed-round step, a twin's own borrowed
# helper). Each is followed and scanned.
SERVED = frozenset('''
attention_grouped attention_replay_audit attention_request_plan dflash_batched_mask dflash_combined_sim_runtime
dflash_pipelined_publish dflash_request_runtime dflash_t16_native_attention dflash_t16_native_scope dflash_traced_publish
draft_convolution_fused draft_kv_history draft_kv_history_tp draft_kv_slide_tp draft_kv_projection draft_kv_projection_tp
draft_kv_projection_trace early_draft extent_attention_replay extent_attention_replay_tp full_dflash_request
gdn_batched_conv gdn_conv_prefix_copy gdn_conv_windows gdn_conv_windows_packed gdn_device_loop_state gdn_records
gdn_seq_block_split gdn_snapshot gdn_vsplit gdn_vsplit_norm_batch gdn_working_state model_link_policy mtp_hidden_capture mtp_hidden_rows ordered_cache
packed_ordered_cache packed_weight_check padded_probe pair_row_exact page_width_tp4 proposal_native_attention
profiled_block_stream_override publish_prewarm quad_draft_tp request_width_warm fused_commit_tp draft_singles_audit draft_convolution_fused_tp draft_shared_head_tp pair_row_exact_tp runtime_binary_override sdpa_tree_scratch serving_gather_experiment
serving_kv_reservation serving_one_in_flight serving_packed_bridge serving_prefill_admission serving_request_quarantine serving_solo_lane serving_fast_lane serving_fast_lane_scheduler stage_profile
target_packed_pages target_t16_attention_8k_gate target_t16_attention_gate tile_collective_tp two_tile_decode two_tile_norm
verifier_engine_tp verifier_position_policy c2_parser_rechunk qwen_prefix_metrics prompt_lookup levern_policy levern_scheduler levern_platform levern_route
tp4_sampdraft tp4_shard_argmax tp4_draft_conv tp4_draft_heads
serving_parked_engines
'''.split())

# Modules the four-card profiles never run their pair code from, and why. The closure does not follow into them.
FLAG_OFF = 'off by the profile env (see the flag)'
NOT_SERVED = {
    'quad_draft': 'the pair quad; the four-card process serves quad_draft_tp in its place (MODULE_TWINS), under QWEN_FAST_QUAD_DRAFT=1',
    'fused_commit': 'the pair fused commit; the four-card process serves fused_commit_tp in its place (MODULE_TWINS), under QWEN_FAST_FUSED_COMMIT=1',
    'draft_kv_slide': 'QWEN_DRAFT_KV_SLIDE_EXPERIMENT=0',
    'draft_kv_slide_scope': 'QWEN_DRAFT_KV_SLIDE_EXPERIMENT=0; scoped_publication is rebound to attach_scopes_tp (TWINS)',
    'mlp_block_stream_pool': 'QWEN_MLP_BLOCK_STREAM_EXPERIMENT=0',
    'mlp_block_stream_request': 'QWEN_MLP_BLOCK_STREAM_EXPERIMENT=0',
    'mlp_block_stream_runtime': 'QWEN_MLP_BLOCK_STREAM_EXPERIMENT=0',
    'gdn_shared_qk_scope': 'QWEN_GDN_SHARED_QK_EXPERIMENT=0',
    'gdn_shared_qk_pipeline': 'QWEN_GDN_SHARED_QK_EXPERIMENT=0',
    'gdn_gate_exp_gate': 'QWEN_GDN_GATE_EXP_ABBA=0',
    'gdn_gate_exp_scope': 'QWEN_GDN_GATE_EXP_ABBA=0',
    'gdn_grouped_gather_gate': 'QWEN_GDN_GROUPED_GATHER_ABBA=0',
    'gdn_grouped_gather_scope': 'QWEN_GDN_GROUPED_GATHER_ABBA=0',
    'gdn_direct_window_scope': 'QWEN_GDN_DIRECT_WINDOW=0; scoped_direct_windows is rebound to attach_scopes_tp (TWINS)',
    'gdn_direct_window_gate': 'QWEN_GDN_DIRECT_WINDOW=0',
    'draft_dot': 'the composed draft attention, which refuses at four cards (draft_attention_tp)',
    'draft_row_sum': 'the composed draft attention, which refuses at four cards (draft_attention_tp)',
    'draft_live_attention': 'the live-query QK arm (QWEN_FAST_LIVE_QUERY_QK), unset in the four-card profiles',
    'draft_live_qk': 'the live-query QK arm (QWEN_FAST_LIVE_QUERY_QK), unset in the four-card profiles',
    'native_draft_sdpa': 'the DSpark native draft SDPA experiment, reached only from measurement scripts',
    'dspark_context_selection': 'the DSpark experiment ladder; not a served path',
    'dspark_score_layout_scope': 'the compact-score experiment scope; entered inert at the attach (test_tp4_attach_profile)',
    'fused_t16_scope': 'the pair fused-T16 experiment scope; entered inert at the attach (test_tp4_attach_profile)',
    'gdn_norm_scatter': 'the shared-QK experiment'"'"'s norm scatter (QWEN_GDN_SHARED_QK_EXPERIMENT=0)',
    'mlp_down_grid_gate': 'the down-grid experiment gate: source hashes, entered inert at the attach',
    'mlp_down_grid_scope': 'the down-grid experiment scope: entered inert at the attach (test_tp4_attach_profile)',
    'mlp_register_epilogue_gate': 'the register-epilogue experiment gate: source hashes, entered inert at the attach',
    'mlp_weight_pipeline_report': 'the weight-pipeline experiment report (a hashed source list)',
    'dflash_t32_native_scope': 'measure_dflash_request, a measurement function of full_dflash_request',
    'full_request': 'measure_dflash_request, a measurement function of full_dflash_request',
    'live_attention_gate': 'measure_dflash_request, a measurement function of full_dflash_request',
    'proposal_native_request': 'measure_dflash_request, a measurement function of full_dflash_request',
    'request_verifier_profile': 'measure_dflash_request, a measurement function of full_dflash_request',
}


# ---- the closure --------------------------------------------------------------------------------------------------

def module_names():
    return {path.stem for path in HERE.glob('*.py')}


def not_served(name):
    return name in NOT_SERVED


def imports_of(name, known):
    """(module-level imports, {lazy module: [functions importing it]}) of scripts/ci/<name>.py, from source text. A function-
    or lambda-level import, and importlib.import_module('name'), counts as lazy."""
    tree = ast.parse((HERE / (name + '.py')).read_text(encoding='utf-8'))
    top, lazy = set(), {}

    def visit(node, depth, function):
        for child in ast.iter_child_nodes(node):
            is_function = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            inner = depth + (1 if is_function or isinstance(child, ast.Lambda) else 0)
            where = child.name if is_function else function
            found = set()
            if isinstance(child, ast.Import):
                found = {alias.name.split('.')[0] for alias in child.names}
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                found = {child.module.split('.')[0]}
            elif (isinstance(child, ast.Call) and getattr(child.func, 'attr', getattr(child.func, 'id', '')) == 'import_module'
                  and child.args and isinstance(child.args[0], ast.Constant) and isinstance(child.args[0].value, str)):
                found = {child.args[0].value.split('.')[0]}
            for module in found & known:
                if inner:
                    lazy.setdefault(module, []).append(where)
                else:
                    top.add(module)
            visit(child, inner, where)
    visit(tree, 0, '')
    return top, lazy


def reach(names, known):
    """Every module reachable from `names` by any import, module-level or lazy."""
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        top, lazy = imports_of(name, known)
        todo += sorted((top | set(lazy)) - seen)
    return seen


# The NOT_SERVED modules the four-card attach really loads (test_tp4_attach_profile enters them inert): the experiment scopes
# and hash gates the attach constructs with their flags off.
ATTACH_ENTERS_INERT = frozenset('''
draft_kv_slide draft_kv_slide_scope fused_t16_scope gdn_direct_window_gate gdn_direct_window_scope gdn_norm_scatter
gdn_shared_qk_pipeline gdn_shared_qk_scope mlp_down_grid_gate mlp_down_grid_scope mlp_register_epilogue_gate
mlp_weight_pipeline_report
'''.split())


def top_level_reach(names, known):
    """Every module reachable from `names` by module-level imports only (what importing them loads)."""
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        top, _ = imports_of(name, known)
        todo += sorted(top - seen)
    return seen


def closure():
    """(modules scanned, {unclassified lazy module: [(importer, function)]}, {pruned module: [importers]})."""
    known = module_names()
    seen, todo, lazy_seen, pruned = set(), list(ROOTS) + sorted(SERVED), {}, {}
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        top, lazy = imports_of(name, known)
        for module in top:
            if not_served(module):
                pruned.setdefault(module, set()).add(name)
            elif module not in seen:
                todo.append(module)
        for module, functions in lazy.items():
            if not_served(module):
                pruned.setdefault(module, set()).add(name)
            else:
                lazy_seen.setdefault(module, []).extend((name, function) for function in functions)
    # A lazy import of a module the closure reaches no other way must have been classified (SERVED, so it is in `seen`,
    # or NOT_SERVED, so it is in `pruned`).
    unclassified = {module: found for module, found in lazy_seen.items() if module not in seen}
    return seen, unclassified, pruned


# ---- the pair's literals ------------------------------------------------------------------------------------------

# Distinctive integers that are a pair width and no other model quantity: the projected qkvzab row (8,240) and its b
# column (8,216), the vocabulary shard (124,160), the MLP shard (8,704) and the state pages (384).
PAIR_INTS = {8240: 'qkvzab row', 8216: 'b column', 124160: 'vocabulary shard', 8704: 'MLP shard', 384: 'GDN state pages'}
# 24 (the pair's GDN value heads, its target query heads) is also an ordinary number, so it counts only where it is a
# shape entry, a heads constant or a head-count comparison.
HEADS_TWENTY_FOUR = 24
CHIP_HINTS = ('shard', 'device', 'chip', 'parts', 'num_devices')


def is_docstring(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def literal_ints(node):
    return tuple(element.value if isinstance(element, ast.Constant) and type(element.value) is int else None
                 for element in node.elts)


def is_range_two(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range'
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 2)


def heads_context(parent):
    """Whether a 24 under `parent` is a head count: a tuple or list of ints (a shape), an assignment to a *HEADS name, or a
    comparison against something named for heads."""
    if isinstance(parent, (ast.Tuple, ast.List)):
        numbers = [element.value for element in parent.elts if isinstance(element, ast.Constant) and type(element.value) is int]
        # 24, 25 and 26 together are circular-buffer ids of the cache kernels, not a shape.
        return len(numbers) >= 2 and 25 not in numbers
    if isinstance(parent, (ast.Assign, ast.AnnAssign)):
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        return any(isinstance(target, ast.Name) and 'HEAD' in target.id.upper() for target in targets)
    if isinstance(parent, ast.Compare):
        return 'head' in ast.unparse(parent).lower()
    return False


def sequences_of(node):
    """The int-or-None rows a shape can be written as: the elements of a tuple or list, and every four consecutive
    positional arguments of a call (`reshape(1, 4, n, 128)`), each as a tuple of ints (None where not a literal int)."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return [literal_ints(node)]
    args = [argument for argument in node.args]
    if any(isinstance(argument, ast.Starred) for argument in args):
        return []
    values = tuple(argument.value if isinstance(argument, ast.Constant) and type(argument.value) is int else None
                   for argument in args)
    if len(values) < 4:
        return [values]
    return [values[start:start + 4] for start in range(len(values) - 3)]


# A pair width that is an ordinary number elsewhere counts only inside a shape: a tuple, list or call of two or more entries. 3072 is the pair's target attention output (12 query heads x 256) and the GDN z width; 8256 the padded qkvzab row.
# (160 and 8192 are not here: 160 x 32 is the hidden size's norm reshape and 8192 a context length, at any width.)
SHAPE_INTS = {3072: 'the pair target attention output / GDN z width', 8256: 'the pair padded qkvzab row'}


def sequence_hits(node):
    """[description]: the pair's shapes in one tuple, list or call: four-wide rows (1, 4, N, 128) K/V banks, (1, 16, N, 128)
    drafter queries, (1, 1, N, 2048/2560) pair-wide queries, (1, 2, N, 256) and (1, 12, N, 256) the target attention's two
    KV and twelve query heads per chip, 2052 rows with the pair KV heads; and SHAPE_INTS inside any row of two or more entries."""
    found = []
    for shape in sequences_of(node):
        if len(shape) == 4:
            if shape[:2] == (1, 4) and shape[3] == 128:
                found.append('(1, 4, N, 128): four KV heads per chip')
            elif shape[:2] == (1, 16) and shape[3] == 128:
                found.append('(1, 16, N, 128): sixteen drafter query heads per chip')
            elif shape[:2] == (1, 1) and shape[3] in (2048, 2560) and shape[2] in (32, 2052, None):
                found.append('(1, 1, N, %d): the pair-wide query or tap' % shape[3])
            elif shape[:2] == (1, 4) and 2052 in shape:
                found.append('2052 rows with the pair KV heads')
            elif shape[:2] == (1, 2) and shape[3] == 256:
                found.append('(1, 2, N, 256): the target attention two KV heads per chip')
            elif shape[:2] == (1, 12) and shape[3] == 256:
                found.append('(1, 12, N, 256): the target attention twelve query heads per chip')
        if len(shape) >= 2:
            found += ['%d (%s)' % (value, SHAPE_INTS[value]) for value in shape if value in SHAPE_INTS]
    return list(dict.fromkeys(found))


def hits_in(source):
    """[(line, qualified function or class, description)]: every pair-only literal in `source`, code only (docstrings and
    comments are not read)."""
    tree = ast.parse(source)
    found, stack = [], []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if is_docstring(child):
                continue
            if (isinstance(child, ast.Constant) and type(child.value) is int and child.value == HEADS_TWENTY_FOUR
                    and heads_context(node)):
                found.append((getattr(child, 'lineno', 0), '.'.join(stack), '24 (GDN value / target query heads)'))
            named = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if named:
                stack.append(child.name)
            where, line = '.'.join(stack), getattr(child, 'lineno', 0)
            if isinstance(child, ast.Constant) and type(child.value) is int and child.value in PAIR_INTS:
                found.append((line, where, '%d (%s)' % (child.value, PAIR_INTS[child.value])))
            elif isinstance(child, (ast.Tuple, ast.List, ast.Call)) and sequence_hits(child):
                for description in sequence_hits(child):
                    found.append((line, where, description))
            elif (isinstance(child, ast.Compare) and len(child.ops) == 1 and isinstance(child.ops[0], (ast.Eq, ast.NotEq))
                  and isinstance(child.comparators[0], ast.Constant) and child.comparators[0].value == 2
                  and any(hint in ast.unparse(child.left) for hint in CHIP_HINTS)):
                found.append((line, where, 'chip count 2: %s' % ast.unparse(child)))
            elif (isinstance(child, (ast.For, ast.comprehension)) and is_range_two(child.iter)
                  and any(hint in ast.unparse(child.target) for hint in CHIP_HINTS)):
                found.append((line, where, 'a chip loop over range(2): %s' % ast.unparse(child.target)))
            elif (isinstance(child, ast.BinOp) and isinstance(child.op, (ast.FloorDiv, ast.Div))
                  and isinstance(child.right, ast.Constant) and child.right.value == 2
                  and re.search(r'5120|hidden|vocab|17408|248320|heads', ast.unparse(child.left))):
                found.append((line, where, 'a width halved for two chips: %s' % ast.unparse(child)))
            visit(child)
            if named:
                stack.pop()
    visit(tree)
    return found


# ---- what is accounted for ---------------------------------------------------------------------------------------

def twinned():
    """{module: {function or class names the tp_addresses seam replaces}} and the modules it replaces whole."""
    import tp_addresses

    names = {}
    for module, name, twin_module, twin_name in tp_addresses.TWINS:
        names.setdefault(module, set()).add(name)
    return names, {original for original, twin in tp_addresses.MODULE_TWINS}


def scan(seen):
    """{module: [(line, where, description)]} for the scanned closure."""
    return {name: hits_in((HERE / (name + '.py')).read_text(encoding='utf-8')) for name in sorted(seen)}


# Hits that stay. Key: (module, the function, or Class.method, that carries the literal - or '<module>'), value: why the
# four-card profiles cannot reach it - or reach it without the pair's number mattering. A key covers that function and what
# is nested in it, nothing else: a new literal in another method of the same class (an inherited one a _tp twin does not
# override) is a new hit and fails. Twin-replaced functions (tp_addresses.TWINS) need no entry.
ALLOWED = {
    ('attention_replay_audit', 'AttentionReplayAudit.check'): 'built only under attention_audit=True, which only the '
                                                             'measurement harnesses pass',
    ('dflash_t16_native_attention', 'numerical_difference'): 'called by the probe script only',
    ('proposal_native_attention', 'numerical_difference'): 'called by the probe script only',
    ('dflash_t16_native_attention_gate', 'qualify'): "checks the pair's recorded simulator report (two chips of the report); "
                                                     "runs inert at the four-card attach (test_tp4_attach_profile)",
    ('live_qk_gate', 'qualify_correctness'): "the pair's recorded qualification report matrix (two chips of the report), "
                                              "not device shapes",
    ('target_t16_attention_gate', 'validate'): "checks the pair's recorded qualification report (two chips of the report)",
    ('target_t16_attention_8k_gate', 'validate'): "checks the pair's recorded qualification report (two chips of the report)",
    ('draft_kv_history', '<module>'): "the pair's bank constants; the four-card class reads tp_shapes (held below)",
    ('draft_kv_history', 'DraftKVHistory.__init__'): "the pair's class: dflash_device and publication_warm build "
                                                     "draft_kv_history_tp at four cards, which overrides this (held below)",
    ('draft_kv_history', 'DraftKVHistory.prepare'): 'overridden by draft_kv_history_tp (held below)',
    ('draft_kv_history', 'DraftKVHistory.audit'): 'overridden by draft_kv_history_tp (held below)',
    ('feature_projection', 'projection_shards'): 'dflash_device selects feature_projection_tp at four cards (held below)',
    ('feature_projection', 'concatenate_local_features'): 'dflash_device selects feature_projection_tp at four cards '
                                                          '(held below)',
    ('full_dflash_request', 'measure_dflash_request'): 'a measurement function; serving_startup uses load_dflash_fixtures only',
    ('full_dflash_request', 'summarize_dflash_requests'): 'a measurement report',
    ('fused_t16_admission', 'qualify_target_weights'): 'runs inside FusedT16Arm, which QWEN_FAST_SINGLE_GATEUP=1 (both '
                                                       'profiles) does not build (dflash_combined_request.combined_runtime)',
    ('packed_weight_check', 'comparison_geometry'): 'called by fused_t16_admission.qualify_target_weights, which is not built '
                                                    '(above)',
    ('packed_weight_check', 'compare_packed_weights'): 'called by fused_t16_admission.qualify_target_weights (above)',
    ('packed_weight_check', 'read_comparison'): 'called by fused_t16_admission.qualify_target_weights (above)',
    ('gdn_user_batch', '<module>'): "the pair's launch constants: gdn_seq_block.batch resolves to gdn_user_batch_tp at four "
                                    "cards (WidthBatch, held below)",
    ('gdn_user_batch', 'validate_users'): "the pair's launch (as above)",
    ('gdn_user_batch', 'execute'): "the pair's launch (as above)",
    ('gdn_vsplit', 'state_page'): "the pair's value-split norm batch: reached only through gdn_batched_conv's "
                                  "run_batched_projected, whose four-card twin refuses use_norm_batch (held below)",
    ('gdn_vsplit', 'bridge_page'): 'the value-split norm batch (as above)',
    ('gdn_vsplit', 'output_element'): 'the value-split norm batch (as above)',
    ('gdn_vsplit', 'stage_spec'): 'the value-split norm batch (as above)',
    ('gdn_vsplit', 'build_program'): 'the value-split norm batch (as above)',
    ('gdn_vsplit', 'execute'): 'the value-split norm batch (as above)',
    ('publication_warm', '<module>'): "the pair's constants; the warm reads kv_shape() and query_shape() (held below)",
    ('serving_buffer_pool', '<module>'): "the pair's constants; the pool reads kv_shape() and query_shape() (held below)",
    ('tp_shapes', '<module>'): 'the geometry table itself: the model constants every width is derived from',
    ('verifier_engine', 'VerifierEngine.verify'): "the pair's sequential engine: verifier_engine_tp.VerifierEngine overrides "
                                                  "verify (the chip-count readback) and serving_request_factory builds it at "
                                                  "four cards (held below)",
    ('verify_trace_t1', '<module>'): "the model's full vocabulary (248,320), the same at any width; the shard width is "
                                     "tp_shapes.vocab_shard",
}


def qualified(where):
    return where or '<module>'


def allowed_key(module, where):
    """The ALLOWED key that covers a hit at `where` (the function or Class.method carrying it, and what is nested in it), or
    None."""
    where = qualified(where)
    for key_module, key in ALLOWED:
        if key_module == module and (where == key or where.startswith(key + '.')):
            return (key_module, key)
    return None


def accounted(module, where, twins, whole):
    top = where.split('.')[0] if where else '<module>'
    return module in whole or top in twins.get(module, ()) or allowed_key(module, where) is not None


PAIR_CONSTANTS = ('KV_SHAPE', 'QUERY_SHAPE')


def constants_read_in_functions(source, names=PAIR_CONSTANTS):
    """[(line, name)]: a pair constant read inside a function body (a module-level definition or a test reading it is not)."""
    found = []
    for function in ast.walk(ast.parse(source)):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found += [(node.lineno, node.id) for node in ast.walk(function)
                      if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names]
    return found


class ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seen, cls.unclassified, cls.pruned = closure()
        cls.hits = scan(cls.seen)
        cls.twins, cls.whole = twinned()

    def test_the_closure_holds_the_modules_the_four_card_attach_and_first_round_import(self):
        for name in ('serving_startup', 'serving_runtime', 'publication_warm', 'dflash_proposal_trace', 'verifier_engine_tp',
                     'draft_kv_history_tp', 'gdn_user_batch_tp', 'tp_shapes', 'tp_addresses', 'feature_projection_tp',
                     'packed_verifier', 'gdn_multitoken_conv', 'extent_attention_replay_tp'):
            self.assertIn(name, self.seen)
        self.assertGreater(len(self.seen), 100)

    def test_every_lazy_import_of_a_served_module_is_classified(self):
        self.assertEqual(self.unclassified, {},
                         'A served module now imports these lazily and they are neither SERVED (scanned) nor NOT_SERVED '
                         '(with the reason the four-card profiles never run them): classify each. importer/function: %r'
                         % {name: found[:3] for name, found in self.unclassified.items()})

    def test_no_pair_literal_on_the_four_card_path_is_unaccounted_for(self):
        loose = []
        for module, found in self.hits.items():
            for line, where, description in found:
                if not accounted(module, where, self.twins, self.whole):
                    loose.append('%s.py:%d in %s: %s' % (module, line, where or '<module>', description))
        self.assertEqual(loose, [],
                         'A pair-only literal sits in a module the four-card serving path imports and no _tp twin '
                         '(tp_addresses.TWINS) replaces its function. Give the number to tp_shapes / a _tp twin, or - if the '
                         'four-card profiles cannot reach it - add ALLOWED with the reason:\n  ' + '\n  '.join(loose))

    def test_the_allowlists_hold_no_stale_entry(self):
        used = set()
        for module, found in self.hits.items():
            for line, where, description in found:
                key = allowed_key(module, where)
                if key is not None:
                    used.add(key)
        self.assertEqual(sorted(set(ALLOWED) - used), [], 'ALLOWED entries that match no literal any more')
        self.assertEqual(sorted(set(NOT_SERVED) - set(self.pruned)), [],
                         'NOT_SERVED entries no served module imports any more')
        self.assertEqual(sorted(SERVED - self.seen), [], 'SERVED entries outside the closure')

    def test_the_fake_attach_imports_nothing_the_closure_did_not_account_for(self):
        """Ground truth for the static closure: run test_tp4_attach_profile (the four-card attach under both profiles, real
        scopes) in a fresh interpreter and take every scripts/ci module it loaded. Each is either scanned here, or sits
        behind a NOT_SERVED module (a pair experiment scope the attach enters inert), so the static model is not missing a
        stage the attach really runs."""
        import json
        import os
        import subprocess

        script = ('import io, json, sys, unittest, contextlib' + chr(10)
                  + 'sys.path.insert(0, %r)' % str(HERE) + chr(10)
                  + 'import test_tp4_attach_profile as t' + chr(10)
                  + 'suite = unittest.defaultTestLoader.loadTestsFromModule(t)' + chr(10)
                  + 'with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):' + chr(10)
                  + '    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)' + chr(10)
                  + 'print(json.dumps(dict(ok=result.wasSuccessful(), modules=sorted(sys.modules))))' + chr(10))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, cwd=str(HERE), timeout=600,
                                env=environment)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace')[-2000:])
        report = json.loads(result.stdout.decode('utf-8').strip().splitlines()[-1])
        self.assertTrue(report['ok'], 'test_tp4_attach_profile fails, so its import set means nothing')
        known = module_names()
        loaded = {name for name in report['modules'] if name in known and not name.startswith('test_')}
        # (make_parked_profiles is the generator of the engine-reuse profile twins: the profile tests that the attach test imports name them; it serves nothing)
        loaded -= {'tp_test_support', 'make_parked_profiles'}
        extra = loaded - self.seen
        # What the attach loads outside the closure is a NOT_SERVED module it enters inert (each named in ATTACH_ENTERS_INERT,
        # exactly) and what those import at module level - not everything any NOT_SERVED module could ever reach.
        self.assertEqual(sorted(extra & set(NOT_SERVED)), sorted(ATTACH_ENTERS_INERT),
                         'the NOT_SERVED modules the four-card attach loads changed: update ATTACH_ENTERS_INERT (and check '
                         'each is entered inert)')
        behind = top_level_reach(sorted(ATTACH_ENTERS_INERT), known)
        self.assertEqual(sorted(extra - behind), [],
                         'the four-card attach imports modules the static closure does not reach: extend ROOTS / SERVED, or '
                         'NOT_SERVED with the reason')

    def test_the_twins_carry_no_pair_literal(self):
        import tp_addresses

        twin_modules = {twin for _, _, twin, _ in tp_addresses.TWINS} | {twin for _, twin in tp_addresses.MODULE_TWINS}
        twin_modules |= {path.stem for path in HERE.glob('*_tp.py') if not path.stem.startswith('test_')}
        twin_modules -= {'tp_addresses'}
        dirty = {}
        for name in sorted(twin_modules):
            found = hits_in((HERE / (name + '.py')).read_text(encoding='utf-8'))
            if found:
                dirty[name] = found
        self.assertEqual(dirty, {}, 'a _tp twin carries the pair literals it exists to remove')


class HeldChoiceTests(unittest.TestCase):
    """The ALLOWED entries that rest on a width-selecting choice hold it in source."""

    def source(self, name):
        return (HERE / (name + '.py')).read_text(encoding='utf-8')

    def test_the_four_card_cache_overrides_what_carries_the_pairs_heads(self):
        tree = ast.parse(self.source('draft_kv_history_tp'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DraftKVHistory')
        self.assertLessEqual({'__init__', 'prepare', 'audit'},
                             {node.name for node in cls.body if isinstance(node, ast.FunctionDef)})

    def test_dflash_device_and_the_warm_choose_the_cache_class_by_width(self):
        for name in ('dflash_device', 'publication_warm'):
            text = self.source(name)
            self.assertIn('from draft_kv_history_tp import DraftKVHistory', text, name)
            self.assertIn('tp_shapes.chip_count() == tp_shapes.PAIR', text, name)

    def test_dflash_device_selects_the_feature_projection_by_width(self):
        text = self.source('dflash_device')
        self.assertIn('feature_projection if tp_shapes.chip_count() == tp_shapes.PAIR else feature_projection_tp', text)

    def test_the_pool_and_the_warm_read_the_served_width_not_their_pair_constants(self):
        for name in ('publication_warm', 'serving_buffer_pool'):
            text = self.source(name)
            self.assertIn('def kv_shape()', text, name)
            self.assertIn('def query_shape()', text, name)
            self.assertEqual(constants_read_in_functions(text), [],
                             '%s reads its pair KV_SHAPE / QUERY_SHAPE inside a function: use kv_shape() / query_shape()' % name)

    def test_the_batched_conv_twin_refuses_the_value_split_norm_batch_at_four_cards(self):
        text = self.source('gdn_batched_conv_tp')
        self.assertIn('if use_norm_batch and tp_shapes.chip_count() != tp_shapes.PAIR:', text)

    def test_the_sequential_engine_is_the_subclass_at_four_cards(self):
        tree = ast.parse(self.source('verifier_engine_tp'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VerifierEngine')
        self.assertIn('verify', {node.name for node in cls.body if isinstance(node, ast.FunctionDef)})
        self.assertIn('from verifier_engine_tp import VerifierEngine', self.source('serving_request_factory'))

    def test_gdn_seq_block_resolves_its_batch_module_by_width(self):
        self.assertIn('return pair_batch if tp_shapes.chip_count() == tp_shapes.PAIR else quad_batch',
                      self.source('gdn_seq_block'))


class ScannerTests(unittest.TestCase):
    """The scanner finds each literal it is for, in code only."""

    def kinds(self, source):
        return [description for _, _, description in hits_in(source)]

    def test_each_kind_of_pair_literal_is_found(self):
        cases = {
            'x = (1, 4, 2048, 128)': '(1, 4, N, 128)',
            'def f():\n    return (1, 4, rows, 128)': '(1, 4, N, 128)',
            'y = (1, 16, 32, 128)': '(1, 16, N, 128)',
            'z = (1, 1, 32, 2048)': '(1, 1, N, 2048)',
            't = (1, 1, 2052, 2560)': '(1, 1, N, 2560)',
            'w = 8240': '8240',
            'w = width - 8216': '8216',
            'v = 124160': '124160',
            'HEADS = 24': '24 (GDN',
            'shape = (1, rows, 24)': '24 (GDN',
            'shape = (rows, 24, 128, 128)': '24 (GDN',
            'if len(shards) != 2:\n    pass': 'chip count 2',
            'for chip in range(2):\n    pass': 'range(2)',
            'half = hidden_size // 2': 'halved',
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertTrue(any(expected in kind for kind in self.kinds(source)), (source, self.kinds(source)))

    def test_list_shapes_call_arguments_and_the_target_attention_widths_are_found(self):
        cases = {
            'x = [1, 4, n, 128]': '(1, 4, N, 128)',
            'y = t.reshape(1, 4, n, 128)': '(1, 4, N, 128)',
            'y = t.reshape(x, 1, 16, n, 128)': '(1, 16, N, 128)',
            'z = t.reshape(1, 1, 32, 2560)': '(1, 1, N, 2560)',
            'k = (1, 2, n, 256)': '(1, 2, N, 256)',
            'q = [1, 12, n, 256]': '(1, 12, N, 256)',
            'o = (1, rows, 3072)': '3072',
            'o = t.reshape(rows, 3072)': '3072',
            'p = (rows, 8256)': '8256',
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertTrue(any(expected in kind for kind in self.kinds(source)), (source, self.kinds(source)))
        self.assertEqual(hits_in('n = 3072' + chr(10) + 'shape = (1, 8, 2048, 128)' + chr(10)), [])

    def test_an_allowed_entry_covers_its_own_method_and_not_a_sibling_method_of_the_class(self):
        self.assertIsNotNone(allowed_key('draft_kv_history', 'DraftKVHistory.prepare'))
        self.assertIsNotNone(allowed_key('draft_kv_history', 'DraftKVHistory.prepare.inner'))
        self.assertIsNone(allowed_key('draft_kv_history', 'DraftKVHistory.upload'))
        self.assertIsNone(allowed_key('draft_kv_history', 'DraftKVHistory.commit'))
        self.assertIsNone(allowed_key('verifier_engine', 'VerifierEngine.other'))
        self.assertFalse([key for key in ALLOWED if key[1] == '*'], 'no whole-module entries')

    def test_prose_and_unrelated_numbers_are_not_found(self):
        source = ('"""(1, 4, 2048, 128) and 8240 in a module docstring."""\n'
                  'def f():\n    """8240 here too."""\n    # (1, 4, 2048, 128) 124160\n'
                  '    return retries == 24 or (1, 8, 2048, 128) == shape[:4] or len(items) != 3 or rows // 4\n')
        self.assertEqual(hits_in(source), [])
        self.assertEqual(hits_in('buffer([24, 25], 16)' + chr(10) + 'args = (0, 1, 24, 25, 26, 16)' + chr(10)), [])

    def test_the_literals_the_second_review_found_are_found_where_they_were(self):
        """publication_warm's bank and query, and dflash_proposal_trace's bank slices, as they stood before the fix."""
        warm = 'def warm():\n    active = zeros((1, 4, 2048, 128))\n    query = zeros((1, 1, 32, 2048))\n'
        trace = 'def update(bank):\n    return operations.slice(bank, (0, 0, 0, 0), (1, 4, context, 128))\n'
        self.assertEqual(len(hits_in(warm)), 2)
        self.assertEqual(len(hits_in(trace)), 1)

    def test_the_pre_fix_publication_warm_and_proposal_capture_fail_the_guard(self):
        """Against the real pre-fix sources (skipped without git history)."""
        import subprocess

        for name in ('publication_warm', 'dflash_proposal_trace'):
            try:
                result = subprocess.run(['git', 'show', '70bbe7ba^:scripts/ci/%s.py' % name], capture_output=True,
                                        cwd=str(HERE), timeout=60)
            except (OSError, subprocess.SubprocessError):
                self.skipTest('no git')
            if result.returncode != 0:
                self.skipTest('no history')
            text = result.stdout.decode('utf-8')
            loose = [hit for hit in hits_in(text) if not accounted(name, hit[1], {}, set())]
            # dflash_proposal_trace's slices sit in functions; publication_warm's sat in module constants the warm read.
            self.assertTrue(loose or constants_read_in_functions(text), name)


if __name__ == '__main__':
    unittest.main()
