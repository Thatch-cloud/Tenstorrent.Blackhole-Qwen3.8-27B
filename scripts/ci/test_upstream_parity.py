"""Parity of the plain-torch DFlash2 port (dflash2_torch.py) and the fine-tune's loss (ft_loss.py) with the UPSTREAM sources they mirror.

The upstream files are pinned and private to the lab (not in this repository), so every test here is skipped unless its environment
variable names the file; CI skips them and a person with the sources runs them (the results go in docs/ft-tau-fine-tune.md):

    ZLAB_SRC=<z-lab model.py>                       the inference model: forward, convolution, selector (needs transformers)
    SPECFORGE_SRC=<the recipe's online model file>  the training objective's terms and the block mask
    SPECFORGE_DFLASH2_SRC=<the recipe's DFlash2 draft file>   the candidate selector's score function

Tiny configurations, float32, CPU; every comparison is a tensor equality within 1e-5 (the same arithmetic in the same order).
"""
import ast
import importlib.util
import os
import sys
import types
import unittest
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import dflash2_torch as d2  # noqa: E402
import ft_loss  # noqa: E402

TOL = dict(atol=1e-5, rtol=1e-5)


def env_file(name):
    path = os.environ.get(name)
    return path if path and os.path.isfile(path) else None


def load_zlab():
    path = env_file('ZLAB_SRC')
    if not path:
        return None
    try:
        import transformers  # noqa: F401
        spec = importlib.util.spec_from_file_location('zlab_model_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def extract(path, class_name, wanted_methods=(), functions=(), classes=()):
    """Execute selected pieces of a source file in a namespace that has torch and typing, WITHOUT importing the file (it imports the
    recipe's package): the named top-level classes and functions, and `wanted_methods` of class `class_name` as plain functions."""
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    import typing
    namespace = dict(torch=torch, nn=nn, F=torch.nn.functional, Optional=typing.Optional, Tuple=typing.Tuple, NamedTuple=typing.NamedTuple,
                     List=typing.List, Dict=typing.Dict, Callable=typing.Callable, Sequence=typing.Sequence)
    methods = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
        if isinstance(node, ast.ClassDef) and node.name in classes:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in wanted_methods:
                    holder = {}
                    exec(compile(ast.Module(body=[item], type_ignores=[]), path, 'exec'), namespace, holder)
                    methods[item.name] = holder[item.name]
    return namespace, methods


# -- the inference model: z-lab model.py ---------------------------------------------------------------------------------------------

ZLAB = load_zlab()
HIDDEN, LAYERS, HEADS, KV, DIM, VOCAB, WINDOW, TAPS, RANK, TOPK, CONV, GROUP = 64, 2, 4, 2, 16, 512, 32, 3, 8, 4, 2, 16


def zlab_model(seed=0):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
    config = Qwen3Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                         num_key_value_heads=KV, head_dim=DIM, rms_norm_eps=1e-6, max_position_embeddings=4096,
                         layer_types=["sliding_attention"] * LAYERS, sliding_window=WINDOW, use_sliding_window=True, attention_bias=False,
                         rope_parameters=dict(rope_type='default', rope_theta=10000.0))
    config.is_causal = False                           # the inference model's block rows see each other both ways (z-lab config)
    config.num_target_layers = 64                      # upstream evaluates build_target_layer_ids(...) eagerly as a default
    config.dflash_config = dict(target_layer_ids=list(range(TAPS)), block_size=8, mask_token_id=VOCAB - 1, selector_rank=RANK,
                                selector_top_k=TOPK, conv_kernel_size=CONV, conv_group_size=GROUP)
    torch.manual_seed(seed)
    model = ZLAB.DFlash2DraftModel(config).float().eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if name.endswith('norm.weight') or name.endswith('layernorm.weight'):
                parameter.copy_(1.0 + 0.1 * torch.randn(parameter.shape, generator=generator))
            elif name.endswith('base_kernel'):
                base = torch.zeros_like(parameter)
                base[:, 0] = 1.0
                parameter.copy_(base + 0.1 * torch.randn(parameter.shape, generator=generator))
            else:
                parameter.copy_(0.05 * torch.randn(parameter.shape, generator=generator))
    return model


def port_cfg():
    return d2.tiny_config(hidden=HIDDEN, layers=LAYERS, heads=HEADS, kv_heads=KV, head_dim=DIM, vocab=VOCAB, window=WINDOW, block=8,
                          taps=TAPS, selector_rank=RANK, top_k=TOPK, intermediate=96, conv_group=GROUP, conv_taps=CONV, mask_token=VOCAB - 1)


def port_from(model):
    """The port with the upstream model's tensors under the port's names (the selector's codebooks are Embeddings upstream)."""
    state = {}
    for name, value in model.state_dict().items():
        state[name.replace('predecessor_codebook.weight', 'predecessor_codebook').replace('successor_codebook.weight', 'successor_codebook')] = value
    port = d2.Dflash2(port_cfg())
    port.load_state_dict(state, strict=True)                  # every tensor name and shape must match, in both directions
    return port


@unittest.skipUnless(ZLAB is not None, 'ZLAB_SRC does not name an importable z-lab model.py (needs transformers)')
class ZlabParityTests(unittest.TestCase):
    def test_the_tensor_names_and_shapes_are_the_same_set(self):
        model = zlab_model()
        port = d2.Dflash2(port_cfg())
        upstream = dict((name.replace('.weight', '') if 'codebook' in name else name, tuple(v.shape)) for name, v in model.state_dict().items())
        ours = dict((name, tuple(v.shape)) for name, v in port.state_dict().items())
        self.assertEqual(upstream, ours)

    def test_the_draft_forward_equals_upstreams_over_a_context_longer_than_the_window(self):
        model = zlab_model()
        port = port_from(model).float().eval()
        generator = torch.Generator().manual_seed(5)
        for context, start in ((40, 40), (12, 100), (70, 70)):
            raw = torch.randn(1, context, TAPS * HIDDEN, generator=generator)
            noise = torch.randn(1, 8, HIDDEN, generator=generator)
            positions = torch.arange(start - context, start + 8)[None]
            with torch.no_grad():
                want = model(position_ids=positions, noise_embedding=noise, target_hidden=raw, past_key_values=None, use_cache=False)
                got = port(noise, positions[:, context:], raw_rows=raw, ctx_positions=positions[:, :context])
            self.assertTrue(torch.allclose(want, got, **TOL), (context, start, float((want - got).abs().max())))

    def test_a_window_shorter_than_the_context_really_cuts_it(self):
        model = zlab_model()
        port = port_from(model).float().eval()
        generator = torch.Generator().manual_seed(6)
        raw = torch.randn(1, 60, TAPS * HIDDEN, generator=generator)
        noise = torch.randn(1, 8, HIDDEN, generator=generator)
        positions = torch.arange(0, 68)[None]
        changed = raw.clone()
        changed[:, :10] += 5.0                                    # rows older than the window: no effect on a block at position 60
        with torch.no_grad():
            first = model(position_ids=positions, noise_embedding=noise, target_hidden=raw)
            second = model(position_ids=positions, noise_embedding=noise, target_hidden=changed)
            mine = port(noise, positions[:, 60:], raw_rows=raw, ctx_positions=positions[:, :60])
            mine_changed = port(noise, positions[:, 60:], raw_rows=changed, ctx_positions=positions[:, :60])
        self.assertTrue(torch.allclose(first, second, **TOL))
        self.assertTrue(torch.allclose(mine, mine_changed, **TOL))

    def test_the_grouped_convolution_equals_upstreams(self):
        model = zlab_model()
        port = port_from(model).float().eval()
        generator = torch.Generator().manual_seed(7)
        hidden = torch.randn(2, 8, HIDDEN, generator=generator)
        upstream, ours = model.layers[0].attention_conv, port.layers[0].attention_conv
        with torch.no_grad():
            want_pre, want_post = upstream.prepare(hidden)
            got_pre, got_post = ours.prepare(hidden)
            self.assertTrue(torch.allclose(want_pre, got_pre, **TOL))
            self.assertTrue(torch.allclose(upstream.finish(hidden, want_post), ours.finish(hidden, got_post), **TOL))

    def test_the_selector_path_equals_upstreams_greedy_proposals(self):
        model = zlab_model()
        port = port_from(model).float().eval()
        head = nn.Linear(HIDDEN, VOCAB, bias=False)
        torch.manual_seed(9)
        nn.init.normal_(head.weight, std=0.3)
        for seed in range(6):
            generator = torch.Generator().manual_seed(100 + seed)
            hidden = torch.randn(1, 7, HIDDEN, generator=generator)
            anchor = torch.randint(0, VOCAB, (1,), generator=generator)
            with torch.no_grad():
                want = model.propose(hidden, anchor, head, 0.0)[0]
                got = port.propose(hidden, anchor, head.weight)
            self.assertEqual(want.tolist(), got.tolist(), seed)

    def test_the_upstream_neutral_scalars_are_what_the_port_assumes(self):
        model = zlab_model()
        self.assertEqual(float(ZLAB._draft_value(model.config, 'input_embedding_scale', 1.0)), 1.0)
        self.assertEqual(float(ZLAB._draft_value(model.config, 'output_multiplier', 1.0)), 1.0)
        self.assertIsNone(ZLAB._draft_value(model.config, 'final_logit_softcapping'))


# -- the training objective: the recipe's own functions -------------------------------------------------------------------------------

SPECFORGE, SPECFORGE_DFLASH2 = env_file('SPECFORGE_SRC'), env_file('SPECFORGE_DFLASH2_SRC')


def recipe_objective():
    """(bound `_dflash_objective_chunk_terms` of the recipe, its selector class) executed from source, or None."""
    if not (SPECFORGE and SPECFORGE_DFLASH2):
        return None
    try:
        namespace, methods = extract(SPECFORGE, 'OnlineDFlashModel', wanted_methods=('_dflash_objective_chunk_terms', '_selector_chunk_terms'),
                                     classes=('SelectorTerms', 'DFlashObjectiveTerms'))
        selector_ns, _ = extract(SPECFORGE_DFLASH2, '', classes=('CandidateSelector',))
        namespace.update(methods)
        return namespace, methods, selector_ns['CandidateSelector']
    except Exception:
        return None


RECIPE = recipe_objective()


@unittest.skipUnless(RECIPE is not None, 'SPECFORGE_SRC and SPECFORGE_DFLASH2_SRC do not both name the recipe sources')
class RecipeObjectiveParityTests(unittest.TestCase):
    BLOCKS, BLOCK = 3, 8

    def fixture(self, seed, alpha=1.0):
        namespace, methods, selector_class = RECIPE
        generator = torch.Generator().manual_seed(seed)
        head = nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            head.weight.copy_(0.3 * torch.randn(head.weight.shape, generator=generator))
        recipe_selector = selector_class(hidden_size=HIDDEN, vocab_size=VOCAB, state_rank=RANK, top_k=TOPK, initializer_range=0.05)
        with torch.no_grad():
            for parameter in recipe_selector.parameters():
                parameter.copy_(0.2 * torch.randn(parameter.shape, generator=generator))
        ours = d2.CandidateSelector(port_cfg())
        with torch.no_grad():
            ours.predecessor_codebook.copy_(recipe_selector.predecessor_codebook)
            ours.successor_codebook.copy_(recipe_selector.successor_codebook)
            ours.hidden_projection.weight.copy_(recipe_selector.hidden_projection.weight)
        shape = (2, self.BLOCKS, self.BLOCK)
        hidden = torch.randn(*shape, HIDDEN, generator=generator)
        # targets drawn from the model's own top-k half the time so both covered and uncovered labels occur
        with torch.no_grad():
            logits = hidden @ head.weight.T
        top = logits.topk(TOPK, dim=-1).indices
        pick = torch.randint(0, TOPK, shape, generator=generator)
        covered = top.gather(-1, pick[..., None])[..., 0]
        random_ids = torch.randint(0, VOCAB, shape, generator=generator)
        target_ids = torch.where(torch.rand(shape, generator=generator) < 0.5, covered, random_ids)
        predecessor_ids = torch.randint(0, VOCAB, shape, generator=generator)
        weight_mask = (torch.rand(shape, generator=generator) > 0.25).float()
        weight_mask[..., 0] = 0.0                                              # the anchor row carries no weight
        fake = SimpleNamespace(lm_head=head, draft_model=SimpleNamespace(candidate_selector=recipe_selector, transform_unary_logits=lambda x: x),
                               loss_type='dflash', loss_decay_gamma=7.0, block_size=self.BLOCK, lk_loss_type=None,
                               _selector_objective_enabled=True, selector_stop_gradient=False)
        fake._selector_chunk_terms = types.MethodType(methods['_selector_chunk_terms'], fake)
        recipe = types.MethodType(methods['_dflash_objective_chunk_terms'], fake)
        return head, recipe_selector, ours, fake, recipe, hidden, target_ids, predecessor_ids, weight_mask

    def test_the_loss_and_every_gradient_equal_the_recipes(self):
        for seed in (1, 2, 3):
            head, recipe_selector, ours, fake, recipe, hidden, target_ids, predecessor_ids, weight_mask = self.fixture(seed)
            h_recipe = hidden.clone().requires_grad_(True)
            terms = recipe(h_recipe, target_ids, weight_mask, predecessor_ids)
            want = (terms.ce_loss_num + 1.0 * terms.selector_ce_num) / terms.loss_den.clamp_min(torch.finfo(terms.loss_den.dtype).tiny)
            want.backward()
            h_ours = hidden.clone().requires_grad_(True)
            got, parts = ft_loss.dflash_loss(h_ours, head.weight.detach(), ours, target_ids, predecessor_ids, weight_mask)
            got.backward()
            self.assertTrue(torch.allclose(want, got, **TOL), (seed, float(want.detach()), float(got.detach())))
            self.assertTrue(torch.allclose(h_recipe.grad, h_ours.grad, **TOL), seed)
            for name in ('predecessor_codebook', 'successor_codebook'):
                self.assertTrue(torch.allclose(getattr(recipe_selector, name).grad, getattr(ours, name).grad, **TOL), (seed, name))
            self.assertTrue(torch.allclose(recipe_selector.hidden_projection.weight.grad, ours.hidden_projection.weight.grad, **TOL), seed)

    def test_the_terms_agree_one_by_one(self):
        head, recipe_selector, ours, fake, recipe, hidden, target_ids, predecessor_ids, weight_mask = self.fixture(4)
        terms = recipe(hidden, target_ids, weight_mask, predecessor_ids)
        _, mine = ft_loss.dflash_loss(hidden, head.weight.detach(), ours, target_ids, predecessor_ids, weight_mask)
        self.assertTrue(torch.allclose(terms.ce_loss_num, mine['ce_num'], **TOL))
        self.assertTrue(torch.allclose(terms.loss_den, mine['den'], **TOL))
        self.assertTrue(torch.allclose(terms.selector_ce_num, mine['selector_num'], **TOL))
        self.assertTrue(torch.allclose(terms.selector_weight_den, mine['covered_den'], **TOL))
        self.assertTrue(torch.allclose(terms.correct_num, mine['correct'], **TOL))

    def test_the_selector_score_function_equals_the_recipes(self):
        head, recipe_selector, ours, fake, recipe, hidden, target_ids, predecessor_ids, weight_mask = self.fixture(5)
        candidates = torch.randint(0, VOCAB, (2, self.BLOCKS, self.BLOCK, TOPK))
        unary = torch.randn(2, self.BLOCKS, self.BLOCK, TOPK)
        want = recipe_selector.score_candidates(candidate_ids=candidates, unary_logits=unary, hidden_states=hidden, predecessor_ids=predecessor_ids)
        got = ours.score_candidates(candidates, unary, hidden, predecessor_ids)
        self.assertTrue(torch.allclose(want, got, **TOL))

    def test_the_chunked_loss_equals_the_recipes_unchunked_one(self):
        head, recipe_selector, ours, fake, recipe, hidden, target_ids, predecessor_ids, weight_mask = self.fixture(6)
        terms = recipe(hidden, target_ids, weight_mask, predecessor_ids)
        want = (terms.ce_loss_num + terms.selector_ce_num) / terms.loss_den
        got, _ = ft_loss.dflash_loss(hidden, head.weight.detach(), ours, target_ids, predecessor_ids, weight_mask, chunk_blocks=1)
        self.assertTrue(torch.allclose(want, got, **TOL))


if __name__ == '__main__':
    unittest.main()
