"""Drafter manifests: the default is today's served drafter bit for bit, a candidate loads under its own pins only.

CPU only. The default manifest is held equal to the UNEDITED loaders' pin tables (draft_*_fixture.py), so the
production drafter's identity cannot drift through the manifest; a candidate is exercised end to end on a tiny synthetic
checkpoint (describe, stage, load) and refused when its fixtures belong to another revision.
"""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import draft_attention_fixture
import draft_convolution_fixture
import draft_mlp_fixture
import draft_projection_fixture
import draft_projection_full_fixture
import draft_remaining_layers_fixture
import draft_selector_fixture
import drafter_fixtures
import drafter_manifest as manifests
import drafter_stage

HERE = Path(__file__).resolve().parent
BF16_HALF = b'\x00\x3f'


def original_tables():
    """Every (tensor name -> (shape, file, sha256)) the unedited loaders pin."""
    tables = {}

    def add(specifications, hashes):
        for name, (shape, filename) in specifications.items():
            tables[name] = (list(shape), filename, hashes[name])

    add(draft_attention_fixture.TENSORS, draft_attention_fixture.TENSOR_SHA256)
    add(draft_convolution_fixture.TENSORS, draft_convolution_fixture.TENSOR_SHA256)
    add(draft_mlp_fixture.TENSORS, draft_mlp_fixture.TENSOR_SHA256)
    add(draft_projection_full_fixture.TENSORS, draft_projection_full_fixture.TENSOR_SHA256)
    add(draft_selector_fixture.TENSORS, draft_selector_fixture.TENSOR_SHA256)
    for layer in (1, 2, 3, 4):
        add(draft_remaining_layers_fixture.specifications(layer),
            draft_remaining_layers_fixture.TENSOR_SHA256[str(layer)])
    return tables


def synthetic_manifest(name='synthetic', revision='a' * 40, fc_columns=3):
    """A manifest of tiny tensors with the layout's names; the constants are the real header and size."""
    tensors, blobs = {}, {}
    for index, tensor in enumerate(sorted(manifests.expected_names())):
        shape = [2, fc_columns] if tensor == 'fc.weight' else [2, 2]
        count = 1
        for dimension in shape:
            count *= dimension
        blob = (BF16_HALF * (count - 1)) + (b'\x80\x3f' if index % 2 else BF16_HALF)
        blobs[tensor] = blob
        tensors[tensor] = dict(shape=shape, dtype='BF16', sha256=hashlib.sha256(blob).hexdigest())
    first32 = hashlib.sha256(blobs['fc.weight'][:32 * fc_columns * 2]).hexdigest()
    manifest = dict(name=name, model='someone/Some-DFlash2', revision=revision, header_sha256=manifests.HEADER_SHA256,
        checkpoint_bytes=manifests.CHECKPOINT_BYTES, config_sha256='b' * 64, fc_first32_sha256=first32,
        trained_block_size=16, tensors=tensors)
    return manifest, blobs


def write_checkpoint(path, blobs, manifest):
    header, offset = {}, 0
    for tensor in sorted(blobs):
        header[tensor] = dict(dtype='BF16', shape=manifest['tensors'][tensor]['shape'],
                              data_offsets=[offset, offset + len(blobs[tensor])])
        offset += len(blobs[tensor])
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(blobs[tensor] for tensor in sorted(blobs)))
    return hashlib.sha256(raw).hexdigest(), path.stat().st_size


class DefaultManifestTests(unittest.TestCase):
    def test_default_equals_the_unedited_loaders(self):
        manifest = manifests.load('dedf8df6')
        tables = original_tables()
        self.assertEqual(set(tables), set(manifest['tensors']))
        self.assertEqual(len(tables), 81)
        for name, (shape, filename, digest) in tables.items():
            self.assertEqual(manifest['tensors'][name], dict(shape=shape, dtype='BF16', sha256=digest), name)
        self.assertEqual(manifest['model'], draft_projection_fixture.MODEL)
        self.assertEqual(manifest['revision'], draft_projection_fixture.REVISION)
        self.assertEqual(manifest['header_sha256'], draft_projection_full_fixture.HEADER_SHA256)
        self.assertEqual(manifest['header_sha256'], manifests.HEADER_SHA256)
        self.assertEqual(manifest['checkpoint_bytes'], 3848817896)
        self.assertEqual(manifest['trained_block_size'], 8)
        source = (HERE / 'draft_projection_full_fixture.py').read_text(encoding='utf-8')
        self.assertIn(manifest['fc_first32_sha256'], source)

    def test_file_layout_matches_the_loaders(self):
        for specifications, files in ((draft_attention_fixture.TENSORS, manifests.ATTENTION),
                                      (draft_convolution_fixture.TENSORS, manifests.CONVOLUTION),
                                      (draft_mlp_fixture.TENSORS, manifests.MLP)):
            self.assertEqual({name.replace('layers.0.', '', 1): filename for name, (shape, filename) in specifications.items()},
                             files)
        self.assertEqual({name: filename for name, (shape, filename) in draft_projection_full_fixture.TENSORS.items()},
                         manifests.PROJECTION)
        self.assertEqual({name: filename for name, (shape, filename) in draft_selector_fixture.TENSORS.items()},
                         manifests.SELECTOR)

    def test_default_manifest_content_is_pinned(self):
        raw = json.loads((manifests.DIRECTORY / 'dedf8df6.json').read_text(encoding='utf-8'))
        self.assertEqual(manifests.canonical_sha256(raw), manifests.DEFAULT_SHA256)
        raw['tensors']['norm.weight']['sha256'] = '0' * 64
        self.assertNotEqual(manifests.canonical_sha256(raw), manifests.DEFAULT_SHA256)

    def test_a_changed_default_refuses_to_load(self):
        with TemporaryDirectory() as directory:
            changed = json.loads((manifests.DIRECTORY / 'dedf8df6.json').read_text(encoding='utf-8'))
            changed['tensors']['norm.weight']['sha256'] = '0' * 64
            (Path(directory) / 'dedf8df6.json').write_text(json.dumps(changed))
            with patch.object(manifests, 'DIRECTORY', Path(directory)):
                with self.assertRaisesRegex(ValueError, 'production pins'):
                    manifests.load('dedf8df6')

    def test_every_committed_manifest_validates(self):
        self.assertIn('dedf8df6', manifests.names())
        for name in manifests.names():
            manifest = manifests.load(name)
            self.assertEqual(manifest['header_sha256'], manifests.HEADER_SHA256)
            self.assertEqual(len(manifest['tensors']), 81)

    def test_b32_is_not_a_candidate(self):
        # Its licence is not a file in the repository; excluded until legal confirms (docs/drafter-arms.md).
        self.assertFalse([name for name in manifests.names() if 'b32' in name])


class ValidationTests(unittest.TestCase):
    def test_refusals(self):
        good, _ = synthetic_manifest()
        manifests.validate(copy.deepcopy(good), 'synthetic')
        cases = []
        for key, value in (('revision', 'abc'), ('header_sha256', '0' * 64), ('checkpoint_bytes', 1),
                           ('config_sha256', 'zz'), ('fc_first32_sha256', '1'), ('model', 'no-owner')):
            case = copy.deepcopy(good)
            case[key] = value
            cases.append(case)
        missing = copy.deepcopy(good)
        del missing['tensors']['norm.weight']
        cases.append(missing)
        extra = copy.deepcopy(good)
        extra['tensors']['unexpected'] = extra['tensors']['norm.weight']
        cases.append(extra)
        wrong = copy.deepcopy(good)
        wrong['tensors']['norm.weight']['dtype'] = 'F32'
        cases.append(wrong)
        badhash = copy.deepcopy(good)
        badhash['tensors']['norm.weight']['sha256'] = 'x'
        cases.append(badhash)
        for case in cases:
            with self.assertRaises(ValueError):
                manifests.validate(case, 'synthetic')
        with self.assertRaises(ValueError):
            manifests.validate(copy.deepcopy(good), 'other-name')

    def test_names_are_plain(self):
        for name in ('../x', 'a/b', '', 'A', '.hidden', 'x' * 65, None):
            with self.assertRaises(ValueError):
                manifests.path_of(name)
        self.assertEqual(manifests.path_of('b16-98759a49').name, 'b16-98759a49.json')


class SelectionTests(unittest.TestCase):
    def test_default_environment_marker_and_conflict(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(manifests.select(root, {}), 'dedf8df6')
            self.assertEqual(manifests.select(None, {}), 'dedf8df6')
            self.assertEqual(manifests.select(root, {manifests.ENVIRONMENT: 'cand'}), 'cand')
            (root / manifests.MARKER).write_text('cand\n')
            self.assertEqual(manifests.select(root, {}), 'cand')
            self.assertEqual(manifests.select(root, {manifests.ENVIRONMENT: 'cand'}), 'cand')
            with self.assertRaisesRegex(ValueError, 'staged for cand'):
                manifests.select(root, {manifests.ENVIRONMENT: 'dedf8df6'})
            (root / manifests.MARKER).write_text('../escape\n')
            with self.assertRaises(ValueError):
                manifests.select(root, {})


class CandidateRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.blobs = synthetic_manifest()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / 'model.safetensors'
        header_sha, size = write_checkpoint(self.checkpoint, self.blobs, self.manifest)
        # The constants a real checkpoint carries are patched to the synthetic file's, for this test only.
        self.patches = [patch.object(manifests, 'HEADER_SHA256', header_sha),
                        patch.object(manifests, 'CHECKPOINT_BYTES', size),
                        patch.object(manifests, 'load', lambda name: self.manifest)]
        self.manifest['header_sha256'], self.manifest['checkpoint_bytes'] = header_sha, size
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def stage(self):
        cache = self.root / 'cache'
        drafter_stage.stage(self.checkpoint, 'synthetic', cache)
        parts, stack = drafter_stage.directories(cache, self.manifest['revision'])
        fixture = self.root / 'fixture'
        fixture.mkdir()
        for component, directory in parts.items():
            shutil.copytree(directory, fixture / component)
        for layer, directory in stack.items():
            shutil.copytree(directory, fixture / f'layer-{layer}')
        return fixture

    def test_describe_matches_the_pins(self):
        described = drafter_stage.describe(self.checkpoint, 'synthetic', self.manifest['model'], self.manifest['revision'])
        self.assertEqual(described['tensors'], self.manifest['tensors'])
        self.assertEqual(described['fc_first32_sha256'], self.manifest['fc_first32_sha256'])
        self.assertEqual(described['header_sha256'], self.manifest['header_sha256'])
        self.assertEqual(described['checkpoint_bytes'], self.manifest['checkpoint_bytes'])

    def test_stage_then_load_returns_the_four_values(self):
        fixture = self.stage()
        manifests_found, layers, projection, selector = drafter_fixtures.load_candidate(fixture, self.manifest)
        self.assertEqual(len(layers), 5)
        self.assertEqual(set(projection), {'fc.weight', 'hidden_norm.weight'})
        self.assertEqual(set(selector), set(manifests.SELECTOR))
        self.assertEqual(len(manifests_found['layers']), 4)
        attention, convolution, mlp = layers[0]
        self.assertEqual(set(attention), {manifests.layer_name(0, s) for s in manifests.ATTENTION})
        for layer in layers[1:]:
            self.assertEqual(set(layer[0]), {manifests.layer_name(0, s) for s in manifests.LAYER})
        self.assertEqual(list(projection['fc.weight'].shape), [2, 3])

    def test_load_routes_by_the_marker_and_never_calls_the_default_loader(self):
        fixture = self.stage()
        (fixture / manifests.MARKER).write_text('synthetic\n')
        def refuse(root):
            raise AssertionError('the default loader must not run for a candidate')
        with patch.object(drafter_fixtures.manifests, 'select', lambda root, environ=None: 'synthetic'),                 contextlib.redirect_stdout(io.StringIO()) as printed:
            found = drafter_fixtures.load(fixture, refuse)
        self.assertEqual(len(found[1]), 5)
        self.assertIn('[DRAFTER_MANIFEST] synthetic in force', printed.getvalue())
        self.assertIn('81 tensors verified', printed.getvalue())

    def test_the_default_prints_nothing_new(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as printed:
            with patch.object(drafter_fixtures.manifests, 'select', lambda root, environ=None: manifests.DEFAULT):
                drafter_fixtures.load(directory, lambda root: 'default')
        self.assertEqual(printed.getvalue(), '')

    def test_default_goes_through_the_unedited_loader(self):
        calls = []
        with TemporaryDirectory() as directory:
            with patch.object(drafter_fixtures.manifests, 'select', lambda root, environ=None: manifests.DEFAULT):
                result = drafter_fixtures.load(directory, lambda root: calls.append(root) or 'default')
        self.assertEqual(result, 'default')
        self.assertEqual(calls, [directory])

    def test_fixtures_of_another_revision_are_refused(self):
        fixture = self.stage()
        other = copy.deepcopy(self.manifest)
        other['revision'] = 'c' * 40
        with self.assertRaisesRegex(ValueError, 'Pinned synthetic manifest required'):
            drafter_fixtures.load_candidate(fixture, other)

    def test_a_changed_byte_is_refused(self):
        fixture = self.stage()
        victim = fixture / 'selector' / 'norm.bf16'
        victim.write_bytes(b'\x80\x3f' * 4)
        with self.assertRaisesRegex(ValueError, 'Audited synthetic content required'):
            drafter_fixtures.load_candidate(fixture, self.manifest)

    def test_a_changed_manifest_pin_is_refused(self):
        fixture = self.stage()
        other = copy.deepcopy(self.manifest)
        other['tensors']['layers.3.mlp.up_proj.weight']['sha256'] = hashlib.sha256(b'other').hexdigest()
        with self.assertRaises(ValueError):
            drafter_fixtures.load_candidate(fixture, other)

    def test_fc_slice_pin_is_enforced(self):
        fixture = self.stage()
        other = copy.deepcopy(self.manifest)
        other['fc_first32_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'first32'):
            drafter_fixtures.load_candidate(fixture, other)

    def test_stage_refuses_a_checkpoint_that_is_not_the_manifests(self):
        other = copy.deepcopy(self.manifest)
        other['tensors']['norm.weight']['sha256'] = hashlib.sha256(b'other').hexdigest()
        with patch.object(manifests, 'load', lambda name: other):
            with self.assertRaisesRegex(ValueError, 'differs from the manifest'):
                drafter_stage.stage(self.checkpoint, 'synthetic', self.root / 'cache2')

    def test_a_failed_stage_leaves_nothing_and_the_same_cache_stages_again(self):
        other = copy.deepcopy(self.manifest)
        other['tensors']['norm.weight']['sha256'] = hashlib.sha256(b'other').hexdigest()
        cache = self.root / 'cache3'
        with patch.object(manifests, 'load', lambda name: other):
            with self.assertRaisesRegex(ValueError, 'differs from the manifest'):
                drafter_stage.stage(self.checkpoint, 'synthetic', cache)
        self.assertEqual([path for path in cache.rglob('*') if path.is_file()], [], 'no tensor file, partial or manifest remains')
        written = drafter_stage.stage(self.checkpoint, 'synthetic', cache)
        self.assertTrue(written)
        for directory in written:
            self.assertTrue((Path(directory) / 'manifest.json').is_file())
            self.assertEqual([path for path in Path(directory).iterdir() if path.name.endswith('.partial')], [])

    def test_stage_never_overwrites(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            drafter_stage.stage(self.checkpoint, 'synthetic', self.root / 'cache')


class CandidateManifestTests(unittest.TestCase):
    def test_b16_is_pinned_to_one_full_revision_with_a_licence_file(self):
        manifest = manifests.load('b16-98759a49')
        self.assertEqual(manifest['model'], '0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16')
        self.assertEqual(manifest['revision'], '98759a4995e4949c462f9a6197f59e015fd82530')
        self.assertEqual(manifest['license'], 'Apache-2.0')
        self.assertEqual(len(manifest['license_file_sha256']), 64)
        self.assertEqual(manifest['trained_block_size'], 16)
        self.assertEqual(manifest['weights_lfs_sha256'],
                         '843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a')

    def test_b16_is_the_default_layout_with_other_weights(self):
        default, candidate = manifests.load('dedf8df6'), manifests.load('b16-98759a49')
        self.assertEqual(candidate['header_sha256'], default['header_sha256'])
        self.assertEqual(candidate['checkpoint_bytes'], default['checkpoint_bytes'])
        for name, entry in candidate['tensors'].items():
            self.assertEqual(entry['shape'], default['tensors'][name]['shape'], name)
        changed = [name for name, entry in candidate['tensors'].items() if entry != default['tensors'][name]]
        # The fine-tune moved the layers, the fc feature projection and the norms; the candidate selector did not move.
        self.assertEqual(len(changed), 70)
        self.assertFalse([name for name in changed if name.startswith('candidate_selector')])
        self.assertIn('fc.weight', changed)
        self.assertNotEqual(candidate['fc_first32_sha256'], default['fc_first32_sha256'])
        self.assertNotEqual(candidate['revision'], default['revision'])
        self.assertNotEqual(candidate['config_sha256'], default['config_sha256'])

    def test_no_private_detail_in_a_committed_manifest(self):
        for path in manifests.DIRECTORY.glob('*.json'):
            text = path.read_text(encoding='utf-8').lower()
            for word in ('/home/', '192.168', '10.10.', 'thatch', 'password', 'token'):
                self.assertNotIn(word, text, path.name)


class WiringTests(unittest.TestCase):
    def test_serving_startup_loads_through_the_manifest_router(self):
        source = (HERE / 'serving_startup.py').read_text(encoding='utf-8')
        self.assertIn("drafter_fixtures.load(paths['fixtures'], load_dflash_fixtures)", source)

    @unittest.skipUnless((HERE / 'build-c2-serving-image.sh').is_file(), 'the build tools belong to a checkout, not the image')
    def test_build_script_selects_by_manifest(self):
        build = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertIn('C2_DRAFTER_MANIFEST', build)
        self.assertIn('DRAFTER_MANIFEST', build)
        self.assertIn('dedf8df68adfb1afeaf7b7480c0a0243108177b4', build)

    @unittest.skipUnless((HERE / 'build-c2-serving-image.sh').is_file(), 'the build tools belong to a checkout, not the image')
    def test_the_build_script_repeats_the_default_manifest_exactly(self):
        # The script cannot read the manifest before it has staged the context, so it repeats the default's identity: held equal here.
        build = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        default = manifests.load('dedf8df6')
        self.assertIn('revision=%s\n' % default['revision'], build)
        self.assertIn('drafter_model=%s\n' % default['model'], build)
        self.assertIn('drafter_config_sha256=%s\n' % default['config_sha256'], build)

    @unittest.skipUnless((HERE / 'build-c2-serving-image.sh').is_file(), 'the build tools belong to a checkout, not the image')
    def test_a_candidate_build_names_the_staging_command_when_its_fixtures_are_missing(self):
        build = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertIn('drafter_stage.py stage', build)
        self.assertIn('C2_DRAFTER_FIXTURES', build)
        self.assertLess(build.index('C2_DRAFTER_FIXTURES'), build.index('cp -al "$fixtures/dflash2-$component-$revision"'))

    def test_the_default_stays_out_of_every_edited_pin_file(self):
        # The loaders and evidence that name the served drafter are not edited by candidates.
        for name in ('draft_projection_fixture.py', 'full_dflash_request.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertNotIn('drafter_manifest', text, name)
        self.assertIn("REVISION = 'dedf8df68adfb1afeaf7b7480c0a0243108177b4'",
                      (HERE / 'draft_projection_fixture.py').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
