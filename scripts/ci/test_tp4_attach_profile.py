"""The four-card S2 profiles attach, end to end, on fakes.

serving_runtime.attach_combined_runtime is the path every fast profile takes, and until this test nothing on CPU drove
it: the review of the first four-card port found that under either TP4 profile it dies at three places no unit test
reached - the sampler-link scope (sampling_link_policy refuses any mesh but (1, 2)), the draft K/V publication scope
(refuses unless QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1, which the profiles turn off) and the direct-window scope (refuses every
16-row projection but the pair's) - and that the two pinned sources it had edited failed the pair's own attach.

Here the environment is the real one - the C2 image's ENV laid under the profile's env by the contract's own
apply_environment - the seam is installed as serving_startup.start installs it, and attach_combined_runtime runs with the
real scopes (the sampler links against a (1, 4) mesh, dflash_combined_request.combined_runtime with its cumulative T16,
down-grid, shared-QK, publication and direct-window scopes). Only what needs a device or an evidence directory that lives
in the image is a fake: the buffer pool, the packed block, the drafter's weights, the lifecycle, the source and evidence
qualifications (their file digests are test_tp2_pins's job), and the model.
"""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch  # noqa: F401 - imported once, before any test patches sys.modules (patch.dict drops modules first imported inside it)

import cumulative_t16_scope  # noqa: F401
import dflash_combined_request  # noqa: F401
import dflash_packed_proposal_coordinator  # noqa: F401
import dflash_prefill_window  # noqa: F401
import dflash_request_runtime  # noqa: F401
import draft_kv_slide_scope  # noqa: F401
import fused_t16_scope  # noqa: F401
import gdn_direct_window_scope  # noqa: F401
import gdn_shared_qk_gate  # noqa: F401
import gdn_shared_qk_scope  # noqa: F401
import gdn_snapshot  # noqa: F401
import mesh_link_policy
import mlp_down_grid_gate
import packed_any_admission  # noqa: F401
import packed_verifier  # noqa: F401
import profiled_block_stream_override  # noqa: F401
import runtime_binary_override  # noqa: F401
import sampling_link_policy
import serving_gather_experiment  # noqa: F401
import serving_packed_step  # noqa: F401
import verifier_engine  # noqa: F401
import serving_c2_contract as contract
import serving_runtime
import tp_addresses
import tp_shapes
from test_c2_packed_tp4_profiles import image_env, profiles

HERE = Path(__file__).resolve().parent
PROFILES = ('c2-packed-tp4', 'c2-packed-tp4-gate', 'c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate-bf16')
RING_DESCRIPTOR = HERE / 'qwen_p150x4_ring_mesh_graph_descriptor.textproto'


def environment(name):
    """The launched process's environment: the image ENV, then the profile's env and mesh, as the boot lays them."""
    environ = dict(image_env())
    contract.apply_environment(dict(profiles()[name], name=name), environ)
    environ['TT_METAL_HOME'] = str(HERE)
    return environ


def config():
    return SimpleNamespace(
        additional_config={'qwen_fast_t16': True, 'qwen_fast_runtime': {}},
        scheduler_config=SimpleNamespace(max_num_seqs=4, async_scheduling=False, scheduler_cls='set'),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(block_size=64, enable_prefix_caching=False),
        lora_config=None, model_config=SimpleNamespace(max_model_len=131328),
        speculative_config=SimpleNamespace(method='dflash', num_speculative_tokens=15,
                                           draft_sample_method='greedy', rejection_sample_method='standard'))


class Recorder(SimpleNamespace):
    pass


class FakeSampling:
    """SamplingGenerator.tt_sampling: what sampler_links reads and overrides."""

    def __init__(self, shape):
        self.mesh_device = SimpleNamespace(shape=list(shape))
        self.num_argmax_gather_links = 1
        self.tt_ccl = SimpleNamespace()


def fake_module(name, **attributes):
    module = ModuleType(name)
    vars(module).update(attributes)
    return module


class Attach:
    """One attach_combined_runtime call under a profile, on fakes; `seen` records what the scopes did."""

    def __init__(self, name, *, seam=True, shape=(1, 4), tp='4', extra_env=()):
        self.name, self.seam, self.shape = name, seam, tuple(shape)
        self.extra_env = dict(extra_env)
        self.seen = dict(links_inside=None, links_before=None, links_after=None, pool=None, engines=[])

    def model(self):
        layers = [SimpleNamespace(is_full_attention=index % 4 == 3, attention=SimpleNamespace(),
                                  feed_forward=SimpleNamespace(weights=SimpleNamespace(w_gate_up=None)))
                  for index in range(64)]
        return SimpleNamespace(mesh_device=SimpleNamespace(shape=list(self.shape)), layers=layers,
                               args=SimpleNamespace(), num_devices=self.shape[1])

    @contextmanager
    def run(self):
        seen, sampling = self.seen, FakeSampling(self.shape)
        model = self.model()
        worker = SimpleNamespace(vllm_config=config(),
                                 model_runner=SimpleNamespace(model=SimpleNamespace(model=[model]), kv_caches=None))

        class Sampler:
            tt_sampling = sampling

            def __init__(self, **options):
                pass

            def set_trace_bucket(self, bucket):
                seen['links_before'] = sampling.num_argmax_gather_links

        class Pool:
            def __init__(self, operations, mesh, **options):
                seen['pool'] = options
                seen['links_inside'] = sampling.num_argmax_gather_links

            def describe(self):
                return {}

            def close(self):
                pass

        class Block:
            extent = True

            def __init__(self, *args, **options):
                seen['engines'].append(options.get('shape'))

            def describe(self):
                return {}

            def close(self):
                pass

        class Weights:
            def __init__(self, *args, **options):
                pass

            def describe(self):
                return {}

            def close(self):
                pass

        class Lifecycle:
            def __init__(self, *args, **options):
                seen['lifecycle'] = True

            def close(self):
                pass

        class Snapshot:
            def __init__(self, *args, **options):
                pass

        windows = dict(report_sha256='windows', passed=True)
        down = dict(report_sha256=mlp_down_grid_gate.REPORT_SHA256)
        native = contextmanager(lambda *args, **kwargs: (yield {}))
        register = contextmanager(lambda *args, **kwargs: (yield dict(constructions=0, calls=0, restored=True)))
        real_audit = mesh_link_policy.audit_descriptor
        environ = environment(self.name)
        environ.update(self.extra_env)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environ, clear=True))
            stack.enter_context(patch.dict(sys.modules, {
                'models.common.sampling.generator': fake_module('generator', SamplingGenerator=Sampler),
                'models.tt_transformers.tt.ccl': fake_module('ccl', TT_CCL=lambda mesh: SimpleNamespace(),
                                                             tt_all_reduce=lambda *args, **kwargs: None)}))
            combined = dflash_combined_request

            for target, name, value in (
                    (serving_runtime, 'ServingBufferPool', Pool), (serving_runtime, 'PreparedDraftWeights', Weights),
                    (serving_runtime, 'FastServingLifecycle', Lifecycle),
                    (serving_runtime, 'ServingCacheOwner', lambda *args: SimpleNamespace(validate=lambda: None,
                                                                                    physical_pages=8208)),
                    (serving_runtime, 'attach_source_check', lambda *args, **kwargs: {}),
                    (serving_runtime, 'dram_line', lambda pool: ''),
                    (serving_runtime, 'register_dram_admission', lambda pool: (lambda: None)),
                    (gdn_snapshot, 'ActiveSnapshot', Snapshot), (packed_verifier, 'PackedVerifierEngine', Block),
                    (serving_packed_step, 'PackedStep', lambda blocks: SimpleNamespace(blocks=blocks)),
                    (packed_any_admission, 'admit', lambda *args, **kwargs: {}),
                    (packed_any_admission, 'admit_pool', lambda *args, **kwargs: None),
                    (packed_any_admission, 'admit_blocks', lambda *args, **kwargs: None),
                    # the image's evidence directories and the runtime tree are not in the repo; their file digests are
                    # test_tp2_pins's, and each scope below is entered for real
                    (combined, 'qualify_windows', lambda *args, **kwargs: windows),
                    (combined, 'qualify_down', lambda *args, **kwargs: down),
                    (combined, 'qualify_norm', lambda *args, **kwargs: dict(report_sha256='norm')),
                    (combined, 'scoped_register_epilogue', register),
                    (runtime_binary_override, 'install', lambda *args, **kwargs: None),
                    (mesh_link_policy, 'audit_descriptor',
                     lambda env, shape, read=None: real_audit(env, shape, read=lambda path: RING_DESCRIPTOR.read_bytes()))):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch.dict(sys.modules, {'dflash_t16_native_scope': fake_module(
                'dflash_t16_native_scope', scoped_native_t16=native)}))
            stack.enter_context(patch('serving_runtime.pindiag', lambda *args, **kwargs: None))
            mesh_link_policy._ring_projection_links.cache_clear()
            stack.callback(mesh_link_policy._ring_projection_links.cache_clear)
            if self.seam:
                installed = tp_addresses.install(os.environ)
                stack.callback(tp_addresses.uninstall)
                seen['installed'] = installed
            attach = serving_runtime.attach_combined_runtime(
                worker, SimpleNamespace(), directory=HERE, runtime_root=HERE,
                fixtures=(None, [], None, None), native_attention_evidence=HERE / 'native',
                block_stream=None, kv_publication_evidence=HERE / 'slide', eos_ids=(1,), cancelled=lambda: False)
            with attach as attached:
                seen['links_after_attach'] = sampling.num_argmax_gather_links
                seen['attached'] = attached
                yield seen
            seen['links_after'] = sampling.num_argmax_gather_links


class AttachTests(unittest.TestCase):
    def test_each_four_card_profile_attaches_on_fakes(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                with Attach(name).run() as seen:
                    audit = seen['attached']['runtime']
                    # the sampler's gather runs at the ring's audited two links while the attach is open, and is put back
                    self.assertEqual(seen['links_after_attach'], 2)
                    # the pair-only scopes stood down instead of refusing or patching the pair's classes
                    self.assertEqual(audit['publication']['enabled'], False)
                    self.assertEqual(audit['publication']['restored'], True)
                    self.assertEqual(audit['target']['direct']['disabled'], 'four-card profile')
                    self.assertEqual(audit['target']['direct']['hits'], 0)
                    self.assertEqual(seen['pool']['extent_replay'], True)
                    self.assertEqual(len(seen['engines']), 1, 'one 64-row block over four seats')
                self.assertEqual(seen['links_after'], seen['links_before'])
                self.assertTrue(seen['lifecycle'])

    def test_the_timed_profile_attaches_with_the_op_profiler_environment(self):
        """The ops-trace arm (ops_profile_plan.PROFILER_ENV) adds the profiler variables to the timed profile: the attach must
        not refuse them (the block stream, which refuses TT_METAL_DEVICE_PROFILER, is off at four cards)."""
        import ops_profile_plan
        with Attach('c2-packed-tp4-speed', extra_env=ops_profile_plan.PROFILER_ENV).run() as seen:
            self.assertEqual(seen['links_after_attach'], 2)
            self.assertEqual(seen['pool']['extent_replay'], True)
            self.assertEqual(os.environ['QWEN_FAST_PROFILE_DUMP_EVERY'], '2')
            self.assertEqual(os.environ['TT_METAL_DEVICE_PROFILER'], '1')
            self.assertEqual(os.environ['QWEN_MLP_BLOCK_STREAM_EXPERIMENT'], '0')

    def test_the_seam_is_what_makes_the_attach_possible(self):
        """The class of bug the review found: the pair's scopes refuse a (1, 4) mesh, and nothing on CPU said so."""
        with self.assertRaisesRegex(ValueError, 'Scoped workaround requires the two-chip mesh'):
            with Attach('c2-packed-tp4', seam=False).run():
                pass

    def test_a_profile_that_turns_a_pair_only_scope_on_is_refused_at_four_cards(self):
        for flag, message in (('QWEN_DRAFT_KV_SLIDE_EXPERIMENT', 'K/V slide publication'),
                              ('QWEN_GDN_DIRECT_WINDOW', 'direct-window hardware batch')):
            with self.subTest(flag=flag):
                attach = Attach('c2-packed-tp4')
                original = environment

                def laid(name, flag=flag):
                    return dict(original(name), **{flag: '1'})

                with patch(__name__ + '.environment', laid):
                    with self.assertRaisesRegex(ValueError, message):
                        with attach.run():
                            pass

    def test_the_pair_keeps_the_pinned_scopes(self):
        """No seam at the pair: the names the attach looks up are the pinned functions themselves."""
        import cumulative_t16_scope
        import draft_kv_slide_scope
        import gdn_direct_window_scope

        self.assertIs(sampling_link_policy.sampler_links.__module__, 'sampling_link_policy')
        self.assertEqual(cumulative_t16_scope.scoped_direct_windows.__module__, 'gdn_direct_window_scope')
        self.assertIs(gdn_direct_window_scope.scoped_direct_windows, cumulative_t16_scope.scoped_direct_windows)
        self.assertEqual(draft_kv_slide_scope.scoped_publication.__module__, 'draft_kv_slide_scope')


class ProfileEnvironmentTests(unittest.TestCase):
    def test_the_environment_the_attach_sees_names_four_cards_end_to_end(self):
        for name in PROFILES:
            environ = environment(name)
            self.assertEqual((environ['QWEN_FAST_TP'], environ['MESH_DEVICE'], environ['QWEN_PROJECTION_LINKS']),
                             ('4', 'P150x4', '2'), name)
            self.assertEqual(environ['QWEN_GDN_DIRECT_WINDOW'], '0')
            self.assertEqual(environ['QWEN_DRAFT_KV_SLIDE_EXPERIMENT'], '0')
            self.assertEqual(tp_shapes.requested_tp(environ), 4)


if __name__ == '__main__':
    unittest.main()
