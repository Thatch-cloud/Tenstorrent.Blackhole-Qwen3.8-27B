"""Another checkpoint of the same architecture (qwen_c2_checkpoint.py, qwen_c2_checkpoints.json, serving_c2_contract.configure_checkpoint, serving_startup.bind_checkpoint,
c2_smoke_check --expect-checkpoint), held on CPU.

The point of all of it: a second checkpoint booted on the converted-weight cache of the first would load the FIRST one's weights without a word (the cache holds one file per
tensor, named by the tensor alone). These tests hold that the base and a selected checkpoint resolve to different cache directories, that the base's own path is exactly what it was
(production keeps its warm cache), that a snapshot whose config, tokenizer or chat template are not the base's is refused, and that the off path changes nothing.
"""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check as smoke_check  # noqa: E402
import qwen_c2_checkpoint as checkpoint  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_startup as startup  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
CATALOG = json.loads((HERE / 'qwen_c2_checkpoints.json').read_text(encoding='utf-8'))
BASE = CATALOG['base']
SWIFT = CATALOG['checkpoints'][0]
BASE_CACHE = '/experiment-cache/weights'
PAIR = {checkpoint.REPO_ENV: SWIFT['repo'], checkpoint.REVISION_ENV: SWIFT['revision']}
BASE_PAIR = {checkpoint.REPO_ENV: BASE['repo'], checkpoint.REVISION_ENV: BASE['revision']}


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Hub(object):
    """A hub directory in a temp dir with a synthetic catalog (small files, their real digests) and one snapshot per entry: the shape of the real thing, none of its bytes."""

    def __init__(self, test):
        self.directory = tempfile.TemporaryDirectory()
        test.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        names = ['config.json', 'tokenizer_config.json', 'chat_template.jinja']
        self.identity = dict((name, ('identity of ' + name).encode()) for name in names)
        weight_map = dict(('t%d' % i, 'model-0000%d-of-00002.safetensors' % (1 + i % 2)) for i in range(6))
        self.index = json.dumps(dict(metadata=dict(total_size=6), weight_map=weight_map)).encode()
        # the same shard names and sizes, other bytes: what a fine-tune of the same architecture is
        self.shards = dict((label, dict(('model-0000%d-of-00002.safetensors' % n, (label + ' shard %d' % n).encode() * 3) for n in (1, 2))) for label in ('basis', 'other'))

        def entry(repo, revision, label):
            return dict(repo=repo, revision=revision, files=dict((name, sha(data)) for name, data in self.identity.items()),
                        weight_map_sha256=checkpoint.weight_map_digest(weight_map), tensors=6, total_size=6,
                        shards=dict((name, dict(size=len(data), sha256=sha(data))) for name, data in self.shards[label].items()))

        self.catalog = dict(about='test', schema=1, identity_files=names, base=entry('org/base', 'a' * 40, 'basis'),
                            checkpoints=[dict(entry('org/other', 'b' * 40, 'other'), served_model='org/other')])
        self.path = self.root / 'catalog.json'
        self.path.write_text(json.dumps(self.catalog), encoding='utf-8')
        self.snapshot('org/base', 'a' * 40, 'basis')
        self.snapshot('org/other', 'b' * 40, 'other')

    def location(self, repo, revision):
        return self.root / 'hub' / checkpoint.hub_dirname(repo)

    def snapshot(self, repo, revision, label):
        base = self.location(repo, revision)
        snap = base / 'snapshots' / revision
        (base / 'blobs').mkdir(parents=True, exist_ok=True)
        snap.mkdir(parents=True, exist_ok=True)
        for name, data in self.identity.items():
            (snap / name).write_bytes(data)
        (snap / 'model.safetensors.index.json').write_bytes(self.index)
        for name, data in self.shards[label].items():
            (snap / name).write_bytes(data)
        return snap

    def snapshots(self):
        """The profile-style pinned list: the base's location in this hub."""
        return ['%s/hub/%s/snapshots/%s' % (self.root, checkpoint.hub_dirname('org/base'), 'a' * 40)]


class CatalogTests(unittest.TestCase):
    def test_the_checked_in_catalog_loads_and_every_checkpoint_equals_the_base(self):
        catalog = checkpoint.load_catalog()
        self.assertEqual(catalog['base']['revision'], '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
        self.assertEqual([item['repo'] for item in catalog['checkpoints']], ['ukisai/Swift-1.5-Qwen3.8-27b'])
        swift = catalog['checkpoints'][0]
        self.assertEqual(swift['revision'], 'b4c84d42903a8646b25857eb2827288d92ed85a4')
        self.assertEqual(swift['files'], catalog['base']['files'], 'config, generation config, tokenizer, vocab, merges, chat template and preprocessors are byte-identical')
        self.assertEqual(sorted(swift['files']), sorted(catalog['identity_files']))
        for field in ('weight_map_sha256', 'tensors', 'total_size'):
            self.assertEqual(swift[field], catalog['base'][field])
        self.assertEqual(sorted(swift['shards']), sorted(catalog['base']['shards']))
        self.assertTrue(all(swift['shards'][name]['size'] == catalog['base']['shards'][name]['size'] for name in swift['shards']))
        self.assertTrue(all(swift['shards'][name]['sha256'] != catalog['base']['shards'][name]['sha256'] for name in swift['shards']), 'the weights are what differs')

    def test_the_catalogs_base_is_the_snapshot_every_profile_pins(self):
        where = checkpoint.base_location(CATALOG)
        profiles = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
        for name, entry in profiles.items():
            with self.subTest(profile=name):
                self.assertTrue(entry['snapshots'] and all(path.endswith('/' + where) for path in entry['snapshots']), entry['snapshots'])

    def test_a_catalog_whose_checkpoint_differs_from_the_base_is_refused_field_by_field(self):
        for change, text in (
                (lambda c: c['checkpoints'][0]['files'].update({'config.json': '0' * 64}), 'config.json'),
                (lambda c: c['checkpoints'][0]['files'].update({'chat_template.jinja': '0' * 64}), 'chat_template.jinja'),
                (lambda c: c['checkpoints'][0]['files'].update({'tokenizer.json': '0' * 64}), 'tokenizer.json'),
                (lambda c: c['checkpoints'][0].update({'weight_map_sha256': '0' * 64}), 'weight_map_sha256'),
                (lambda c: c['checkpoints'][0].update({'tensors': 1}), 'tensors'),
                (lambda c: c['checkpoints'][0].update({'total_size': 1}), 'total_size'),
                (lambda c: c['checkpoints'][0]['shards']['model-00001-of-00018.safetensors'].update({'size': 1}), 'shards'),
                (lambda c: c['checkpoints'][0]['shards'].pop('model-00018-of-00018.safetensors'), 'shards'),
                (lambda c: c['checkpoints'].append(copy.deepcopy(c['checkpoints'][0])), 'twice'),
                (lambda c: c['checkpoints'][0].update({'revision': 'main'}), 'revision')):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                broken = copy.deepcopy(CATALOG)
                change(broken)
                path = Path(tmp) / 'c.json'
                path.write_text(json.dumps(broken), encoding='utf-8')
                with self.assertRaises(checkpoint.CheckpointError) as raised:
                    checkpoint.load_catalog(str(path))
                self.assertIn(text, str(raised.exception))

    def test_the_contracts_name_list_is_the_modules(self):
        self.assertEqual(set(contract.CHECKPOINT_ENV), {checkpoint.REPO_ENV, checkpoint.REVISION_ENV} | set(checkpoint.DERIVED_ENV))


class SelectionTests(unittest.TestCase):
    def test_nothing_asked_is_none(self):
        self.assertIsNone(checkpoint.selection({}))
        self.assertIsNone(checkpoint.selection({checkpoint.REPO_ENV: '', checkpoint.REVISION_ENV: '  '}))

    def test_one_name_alone_is_refused(self):
        for environ in ({checkpoint.REPO_ENV: SWIFT['repo']}, {checkpoint.REVISION_ENV: SWIFT['revision']}):
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.selection(environ)

    def test_only_a_listed_full_revision_is_taken(self):
        for revision in ('main', SWIFT['revision'][:12], SWIFT['revision'].upper(), 'c' * 40, '', SWIFT['revision'] + '0'):
            with self.subTest(revision=revision), self.assertRaises(checkpoint.CheckpointError):
                checkpoint.selection({checkpoint.REPO_ENV: SWIFT['repo'], checkpoint.REVISION_ENV: revision})
        for repo in ('Swift', 'ukisai/other', '../x/y', 'a/b/c'):
            with self.subTest(repo=repo), self.assertRaises(checkpoint.CheckpointError):
                checkpoint.selection({checkpoint.REPO_ENV: repo, checkpoint.REVISION_ENV: SWIFT['revision']})

    def test_the_base_pair_is_the_base_and_swift_is_not(self):
        base, swift = checkpoint.selection(BASE_PAIR), checkpoint.selection(PAIR)
        self.assertTrue(base.is_base)
        self.assertFalse(swift.is_base)
        self.assertEqual(swift.key, 'ukisai--Swift-1.5-Qwen3.8-27b--b4c84d42903a8646b25857eb2827288d92ed85a4')
        self.assertEqual(checkpoint.served_model(SWIFT['repo'], SWIFT['revision']), 'ukisai/Swift-1.5-Qwen3.8-27b')
        self.assertEqual(checkpoint.served_model(BASE['repo'], BASE['revision']), '')


class CacheDirectoryTests(unittest.TestCase):
    """THE property: two checkpoints never share converted weights, and the base's path is unchanged."""

    def test_the_base_and_swift_resolve_to_different_cache_directories_and_the_base_is_unchanged(self):
        base, swift = checkpoint.selection(BASE_PAIR), checkpoint.selection(PAIR)
        self.assertEqual(checkpoint.keyed_cache_dir(BASE_CACHE, None), BASE_CACHE)
        self.assertEqual(checkpoint.keyed_cache_dir(BASE_CACHE, base), BASE_CACHE, 'production keeps its warm cache: the very same directory')
        keyed = checkpoint.keyed_cache_dir(BASE_CACHE, swift)
        self.assertNotEqual(keyed, BASE_CACHE)
        self.assertEqual(keyed, '/experiment-cache/weights-ckpt/ukisai--Swift-1.5-Qwen3.8-27b--b4c84d42903a8646b25857eb2827288d92ed85a4')
        self.assertFalse(checkpoint.within(keyed, BASE_CACHE), 'not under the base\'s tree: nothing the base reads or writes is reachable from it')
        self.assertFalse(checkpoint.within(BASE_CACHE, keyed))
        for tail in ('P150x4/tensor_cache_bfp8', 'P150x4/tensor_cache_bfp8/visual.merger.tensorbin'):
            self.assertNotEqual(os.path.join(BASE_CACHE, tail), os.path.join(keyed, tail))

    def test_two_revisions_of_one_repo_and_two_repos_never_share(self):
        keys = set()
        for repo, revision in (('ukisai/Swift-1.5-Qwen3.8-27b', 'a' * 40), ('ukisai/Swift-1.5-Qwen3.8-27b', 'b' * 40), ('ukisai/other', 'a' * 40), ('Qwen/Qwen3.8-27B', 'a' * 40)):
            keys.add(checkpoint.cache_key(repo, revision))
        self.assertEqual(len(keys), 4)

    def test_an_empty_base_cache_cannot_be_keyed(self):
        for value in ('', '/', None):
            with self.subTest(value=value), self.assertRaises(checkpoint.CheckpointError):
                checkpoint.keyed_cache_dir(value, checkpoint.selection(PAIR))

    def test_the_environment_of_the_base_is_untouched_and_swift_is_keyed(self):
        for environ in ({'TT_CACHE_PATH': BASE_CACHE}, dict(BASE_PAIR, TT_CACHE_PATH=BASE_CACHE)):
            before = dict(environ)
            self.assertIsNone(checkpoint.configure_environment(environ, checkpoint.selection(environ), None))
            self.assertEqual(environ, before, 'the base: not one variable changed')
        environ = dict(PAIR, TT_CACHE_PATH=BASE_CACHE)
        facts = checkpoint.configure_environment(environ, checkpoint.selection(environ), '/snap')
        self.assertEqual(environ['TT_CACHE_PATH'], checkpoint.keyed_cache_dir(BASE_CACHE, checkpoint.selection(PAIR)))
        self.assertEqual(environ[checkpoint.CACHE_BASE_ENV], BASE_CACHE)
        self.assertEqual((environ[checkpoint.KEY_ENV], environ[checkpoint.SNAPSHOT_ENV]), (checkpoint.selection(PAIR).key, '/snap'))
        self.assertEqual(facts['weight_cache'], environ['TT_CACHE_PATH'])

    def test_a_child_process_that_inherits_the_keyed_environment_derives_the_same_values(self):
        environ = dict(PAIR, TT_CACHE_PATH=BASE_CACHE)
        checkpoint.configure_environment(environ, checkpoint.selection(environ), '/snap')
        child = dict(environ)
        again = checkpoint.configure_environment(child, checkpoint.selection(child), '/snap')
        self.assertEqual(child, environ, 'idempotent: no double keying')
        self.assertEqual(again['weight_cache'], environ['TT_CACHE_PATH'])

    def test_a_moved_or_already_keyed_cache_path_is_refused(self):
        environ = dict(PAIR, TT_CACHE_PATH=BASE_CACHE)
        checkpoint.configure_environment(environ, checkpoint.selection(environ), '/snap')
        moved = dict(environ, TT_CACHE_PATH='/elsewhere')
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.configure_environment(moved, checkpoint.selection(moved), '/snap')
        keyed_only = dict(PAIR, TT_CACHE_PATH=environ['TT_CACHE_PATH'])
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.configure_environment(keyed_only, checkpoint.selection(keyed_only), '/snap')
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.configure_environment(dict(PAIR), checkpoint.selection(PAIR), '/snap')

    def test_the_loaders_variables_may_not_point_at_another_checkpoint(self):
        for name in checkpoint.LOADER_ENV:
            bad = dict(PAIR, TT_CACHE_PATH=BASE_CACHE, **{name: '/models/models--Qwen--Qwen3.8-27B/snapshots/' + BASE['revision']})
            with self.subTest(name=name), self.assertRaises(checkpoint.CheckpointError):
                checkpoint.configure_environment(bad, checkpoint.selection(bad), '/snap')
            good = dict(PAIR, TT_CACHE_PATH=BASE_CACHE, **{name: '/snap'})
            checkpoint.configure_environment(good, checkpoint.selection(good), '/snap')

    def test_the_base_run_drops_a_stale_derived_name(self):
        environ = dict(TT_CACHE_PATH=BASE_CACHE, **{checkpoint.KEY_ENV: 'x', checkpoint.SNAPSHOT_ENV: '/y', checkpoint.CACHE_BASE_ENV: '/z'})
        checkpoint.configure_environment(environ, None, None)
        self.assertEqual(environ, {'TT_CACHE_PATH': BASE_CACHE})


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.hub = Hub(self)
        self.catalog = checkpoint.load_catalog(str(self.hub.path))
        self.sel = checkpoint.selection({checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'b' * 40}, self.catalog)
        self.snapshot = str(self.hub.location('org/other', 'b' * 40) / 'snapshots' / ('b' * 40))

    def verify(self):
        return checkpoint.verify_snapshot(self.snapshot, self.sel, self.catalog)

    def test_a_snapshot_with_the_bases_identity_and_whole_shards_passes(self):
        self.assertEqual(self.verify(), [])

    def test_each_identity_file_is_held_to_the_bases_digest(self):
        for name in self.catalog['identity_files']:
            with self.subTest(name=name):
                path = Path(self.snapshot) / name
                saved = path.read_bytes()
                path.write_bytes(saved + b' ')
                problems = self.verify()
                path.write_bytes(saved)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(name, problems[0])
        (Path(self.snapshot) / 'config.json').unlink()
        self.assertIn('config.json is missing', ' '.join(self.verify()))

    def test_the_index_the_shards_and_a_running_download_are_checked(self):
        index = Path(self.snapshot) / 'model.safetensors.index.json'
        document = json.loads(index.read_text())
        document['weight_map']['t0'] = 'model-00002-of-00002.safetensors'
        index.write_text(json.dumps(document))
        self.assertTrue(any('weight_map' in problem for problem in self.verify()))
        index.write_bytes(self.hub.index)
        shard = Path(self.snapshot) / 'model-00001-of-00002.safetensors'
        saved = shard.read_bytes()
        shard.write_bytes(saved[:-1])
        self.assertTrue(any('bytes' in problem for problem in self.verify()))
        shard.write_bytes(saved)
        shard.unlink()
        self.assertTrue(any('missing' in problem for problem in self.verify()))
        shard.write_bytes(saved)
        (self.hub.location('org/other', 'b' * 40) / 'blobs' / 'abc.incomplete').write_bytes(b'')
        self.assertTrue(any('download' in problem for problem in self.verify()))

    def test_hashing_the_shards_is_the_operators_option(self):
        self.assertEqual(checkpoint.verify_snapshot(self.snapshot, self.sel, self.catalog, hash_shards=True), [])
        shard = Path(self.snapshot) / 'model-00001-of-00002.safetensors'
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(bytes(data))
        self.assertEqual(self.verify(), [], 'the boot check reads sizes only: a flipped byte of the right size passes it')
        self.assertEqual(checkpoint.verify_snapshot(self.snapshot, self.sel, self.catalog, hash_shards=True), ['model-00001-of-00002.safetensors does not hash to its pinned sha256'])

    def test_the_profiles_pinned_snapshot_is_moved_to_the_checkpoints_hub_directory_under_the_same_roots(self):
        pinned = ['/models/models--Qwen--Qwen3.8-27B/snapshots/' + BASE['revision'], '/models/hub/models--Qwen--Qwen3.8-27B/snapshots/' + BASE['revision']]
        sel = checkpoint.selection(PAIR)
        self.assertEqual(checkpoint.snapshot_candidates(pinned, sel, CATALOG), [
            '/models/models--ukisai--Swift-1.5-Qwen3.8-27b/snapshots/' + SWIFT['revision'], '/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27b/snapshots/' + SWIFT['revision']])
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.snapshot_candidates(['/models/models--Qwen--Other/snapshots/' + BASE['revision']], sel, CATALOG)
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.resolve_snapshot(pinned, sel, CATALOG, exists=lambda path: False)


class MarkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sel = checkpoint.selection(PAIR)
        self.environ = dict(PAIR, TT_CACHE_PATH=os.path.join(self.tmp.name, 'weights'))
        checkpoint.configure_environment(self.environ, self.sel, '/snap')

    def test_the_first_boot_creates_the_marker_and_the_worker_completes_it(self):
        claim = checkpoint.claim_cache(self.environ, self.sel)
        self.assertEqual(claim, dict(created=True, first_boot_complete=False))
        marker = Path(self.environ['TT_CACHE_PATH']) / checkpoint.MARKER
        self.assertEqual(json.loads(marker.read_text())['revision'], SWIFT['revision'])
        self.assertTrue(checkpoint.mark_first_boot_complete(self.environ))
        self.assertFalse(checkpoint.mark_first_boot_complete(self.environ), 'already complete')
        self.assertEqual(checkpoint.claim_cache(self.environ, self.sel), dict(created=False, first_boot_complete=True))

    def test_a_directory_whose_marker_names_another_pair_is_refused(self):
        checkpoint.claim_cache(self.environ, self.sel)
        marker = Path(self.environ['TT_CACHE_PATH']) / checkpoint.MARKER
        for field, value in (('revision', 'c' * 40), ('repo', 'org/other'), ('weight_map_sha256', '0' * 64), ('key', 'x')):
            document = json.loads(marker.read_text())
            saved = dict(document)
            document[field] = value
            marker.write_text(json.dumps(document))
            with self.subTest(field=field), self.assertRaises(checkpoint.CheckpointError):
                checkpoint.claim_cache(self.environ, self.sel)
            marker.write_text(json.dumps(saved))

    def test_without_a_selection_the_worker_marks_nothing(self):
        self.assertFalse(checkpoint.mark_first_boot_complete({'TT_CACHE_PATH': self.tmp.name}))


class Args(object):
    def __init__(self, checkpoint_dir, cache):
        self.checkpoint_dir, self._cache = checkpoint_dir, cache

    def weight_cache_path(self, dtype=None):
        return Path(self._cache) / 'P150x4' / 'tensor_cache_bfp8'


class WorkerBindingTests(unittest.TestCase):
    """serving_startup.bind_checkpoint: after the model exists, before the fast path attaches: fail closed."""

    def environ(self):
        environ = dict(PAIR, TT_CACHE_PATH=BASE_CACHE)
        checkpoint.configure_environment(environ, checkpoint.selection(environ), '/models/snap')
        return environ

    def test_off_returns_at_the_first_line(self):
        self.assertFalse(startup.bind_checkpoint(object(), {'target_snapshot': Path('/x')}, environ={'TT_CACHE_PATH': BASE_CACHE}))

    def test_the_selected_snapshot_and_keyed_cache_bind(self):
        environ = self.environ()
        said = []
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / 'snap'
            snap.mkdir()
            environ[checkpoint.SNAPSHOT_ENV] = str(snap)
            args = Args(str(snap), environ['TT_CACHE_PATH'])
            model = type('Model', (), dict(args=args))()
            self.assertTrue(startup.bind_checkpoint(model, {'target_snapshot': snap.resolve()}, environ=environ, say=said.append))
        self.assertEqual(len(said), 1)
        self.assertTrue(said[0].startswith('[QWEN-C2] checkpoint bound %s@%s ' % (SWIFT['repo'], SWIFT['revision'])), said)
        self.assertIn('-ckpt/' + checkpoint.selection(PAIR).key, said[0])

    def test_the_bases_weights_or_the_bases_cache_stop_the_worker(self):
        environ = self.environ()
        with tempfile.TemporaryDirectory() as tmp:
            snap, other = Path(tmp) / 'snap', Path(tmp) / 'other'
            snap.mkdir()
            other.mkdir()
            environ[checkpoint.SNAPSHOT_ENV] = str(snap)
            for label, args, target in (('the model loaded another snapshot', Args(str(other), environ['TT_CACHE_PATH']), snap),
                                        ('the model reads the base cache', Args(str(snap), BASE_CACHE), snap),
                                        ('the engine targets another snapshot', Args(str(snap), environ['TT_CACHE_PATH']), other)):
                with self.subTest(label), self.assertRaises(ValueError) as raised:
                    startup.bind_checkpoint(type('Model', (), dict(args=args))(), {'target_snapshot': target.resolve()}, environ=environ, say=lambda line: None)
                self.assertIn('not the selected checkpoint', str(raised.exception))
            silent = type('Silent', (), {})()          # no checkpoint_dir, no weight_cache_path: cannot be tied, so stop
            with self.assertRaises(ValueError):
                startup.bind_checkpoint(type('Model', (), dict(args=silent))(), {'target_snapshot': snap.resolve()}, environ=environ, say=lambda line: None)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.hub = Hub(self)
        patcher = mock.patch.object(checkpoint, 'DATA', str(self.hub.path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.profile = dict(name='p', snapshots=self.hub.snapshots())
        self.catalog = checkpoint.load_catalog()

    def configure(self, environ, api_server=True):
        lines = []
        with mock.patch.object(contract, 'log', lambda message, *values: lines.append(message % values if values else message)):
            return contract.configure_checkpoint(self.profile, environ, api_server), lines

    def test_nothing_asked_changes_nothing_and_imports_nothing(self):
        environ = {'TT_CACHE_PATH': BASE_CACHE, 'HF_MODEL': 'Qwen/Qwen3.8-27B'}
        before = dict(environ)
        with mock.patch.dict(sys.modules):
            sys.modules.pop('qwen_c2_checkpoint', None)
            self.assertEqual(self.configure(environ), (None, []))
            self.assertNotIn('qwen_c2_checkpoint', sys.modules, 'the off path does not even import the module')
        self.assertEqual(environ, before)

    def test_the_base_pair_moves_nothing(self):
        environ = {'TT_CACHE_PATH': BASE_CACHE, checkpoint.REPO_ENV: 'org/base', checkpoint.REVISION_ENV: 'a' * 40}
        before = dict(environ)
        snapshot, lines = self.configure(environ)
        self.assertIsNone(snapshot)
        self.assertEqual(environ, before)
        self.assertEqual(len(lines), 1)
        self.assertNotIn('[QWEN-C2] checkpoint', '[QWEN-C2] ' + lines[0], 'a base run logs no checkpoint line')

    def test_another_checkpoint_serves_its_snapshot_from_a_keyed_cache_and_says_so(self):
        environ = {'TT_CACHE_PATH': os.path.join(str(self.hub.root), 'weights'), checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'b' * 40}
        base_cache = environ['TT_CACHE_PATH']
        snapshot, lines = self.configure(environ)
        self.assertEqual(snapshot, '%s/hub/models--org--other/snapshots/%s' % (self.hub.root, 'b' * 40))
        self.assertEqual(environ['TT_CACHE_PATH'], base_cache + '-ckpt/org--other--' + 'b' * 40)
        self.assertTrue(os.path.isfile(os.path.join(environ['TT_CACHE_PATH'], checkpoint.MARKER)))
        self.assertFalse(os.path.exists(base_cache), 'the base\'s cache directory is not even created')
        self.assertTrue(lines[0].startswith('checkpoint org/other@' + 'b' * 40 + ' key=org--other--' + 'b' * 40), lines)
        self.assertIn('(the base cache is not read or written)', lines[0])
        text = '\n'.join('[QWEN-C2] ' + line for line in lines)
        self.assertNotIn('Traceback', text)

    def test_only_the_api_server_checks_and_claims(self):
        environ = {'TT_CACHE_PATH': os.path.join(str(self.hub.root), 'weights'), checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'b' * 40}
        (Path(self.hub.location('org/other', 'b' * 40)) / 'snapshots' / ('b' * 40) / 'config.json').write_bytes(b'another config')
        snapshot, lines = self.configure(environ, api_server=False)
        self.assertTrue(snapshot, 'a child process only sets its environment (cheap): the API server refuses')
        self.assertEqual(lines, [])
        self.assertFalse(os.path.exists(environ['TT_CACHE_PATH']))
        environ = {'TT_CACHE_PATH': os.path.join(str(self.hub.root), 'weights'), checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'b' * 40}
        with self.assertRaises(ValueError) as raised:
            self.configure(environ, api_server=True)
        self.assertIn('config.json differs from the base', str(raised.exception))
        self.assertFalse(os.path.exists(environ['TT_CACHE_PATH']), 'a refused snapshot creates no cache directory')

    def test_a_checkpoint_that_is_not_mounted_or_not_listed_refuses_the_boot(self):
        environ = {'TT_CACHE_PATH': BASE_CACHE, checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'c' * 40}
        with self.assertRaises(ValueError):
            self.configure(environ)
        environ = {'TT_CACHE_PATH': BASE_CACHE, checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'b' * 40}
        self.profile['snapshots'] = [str(self.hub.root / 'nowhere' / 'models--org--base' / 'snapshots' / ('a' * 40))]
        with self.assertRaises(ValueError) as raised:
            self.configure(environ)
        self.assertIn('none of the snapshots', str(raised.exception))

    def test_boot_in_the_api_server_serves_the_checkpoints_snapshot_in_the_engine_argv(self):
        """contract.boot end to end with a one-profile file whose snapshots are the temp hub's: --model and target_snapshot are the other checkpoint's snapshot."""
        document = json.loads(PROFILES.read_text(encoding='utf-8'))
        entry = copy.deepcopy(document['profiles']['general-tp4'])
        entry['snapshots'] = self.hub.snapshots()
        profiles = self.hub.root / 'profiles.json'
        profiles.write_text(json.dumps(dict(default='general-tp4', profiles=dict(document['profiles'], **{'general-tp4': entry}))), encoding='utf-8')
        for pair, expected, asked in ((dict(PAIR_OTHER), 'models--org--other', 'org/other'), ({}, 'models--org--base', 'Qwen/Qwen3.8-27B')):
            platform = ['api_server', '--model', asked, '--port', '8000']
            environ = dict(pair, QWEN_C2_SERVING='1', QWEN_C2_PROFILE='general-tp4', QWEN_C2_PROFILES=str(profiles), MESH_DEVICE='P300',
                           TT_CACHE_PATH=os.path.join(str(self.hub.root), 'weights'))
            before, saved = list(sys.meta_path), list(sys.argv)
            sys.argv[:] = platform
            try:
                with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': 'general-tp4'}), mock.patch.object(contract, 'fix_sys_path'), mock.patch.object(contract, 'install_teardown_skip'), \
                        mock.patch.object(contract, 'read_salt_key', return_value=(None, 'none')), mock.patch.object(contract, 'log'):
                    contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
                argv = list(sys.argv)
            finally:
                sys.argv[:] = saved
                sys.meta_path[:] = before
            model = argv[argv.index('--model') + 1]
            self.assertIn(expected, model, pair)
            if pair:
                self.assertIn('-ckpt/', environ['TT_CACHE_PATH'])
            else:
                self.assertEqual(environ['TT_CACHE_PATH'], os.path.join(str(self.hub.root), 'weights'), 'the base: the cache path is exactly what it was')

    def test_boot_refuses_a_platform_that_asks_for_another_model_than_the_image_serves(self):
        document = json.loads(PROFILES.read_text(encoding='utf-8'))
        entry = copy.deepcopy(document['profiles']['general-tp4'])
        entry['snapshots'] = self.hub.snapshots()
        profiles = self.hub.root / 'profiles.json'
        profiles.write_text(json.dumps(dict(default='general-tp4', profiles=dict(document['profiles'], **{'general-tp4': entry}))), encoding='utf-8')
        for pair, asked in ((dict(PAIR_OTHER), 'org/base'), ({}, 'org/other')):
            environ = dict(pair, QWEN_C2_SERVING='1', QWEN_C2_PROFILE='general-tp4', QWEN_C2_PROFILES=str(profiles), MESH_DEVICE='P300', TT_CACHE_PATH=os.path.join(str(self.hub.root), 'weights'))
            before, saved = list(sys.meta_path), list(sys.argv)
            sys.argv[:] = ['api_server', '--model', asked, '--port', '8000']
            try:
                with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': 'general-tp4'}), mock.patch.object(contract, 'fix_sys_path'), mock.patch.object(contract, 'install_teardown_skip'), \
                        mock.patch.object(contract, 'read_salt_key', return_value=(None, 'none')), mock.patch.object(contract, 'log'):
                    with self.subTest(pair=pair, asked=asked), self.assertRaises(ValueError) as raised:
                        contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
                    self.assertIn('the platform asks for the model', str(raised.exception))
            finally:
                sys.argv[:] = saved
                sys.meta_path[:] = before


PAIR_OTHER = {checkpoint.REPO_ENV: 'org/other', checkpoint.REVISION_ENV: 'b' * 40}


class LogReadTests(unittest.TestCase):
    KEY = checkpoint.selection(PAIR).key
    HEAD = '[QWEN-C2] checkpoint %s@%s key=%s snapshot=/s weight_cache=/experiment-cache/weights-ckpt/%s base_cache=/experiment-cache/weights (the base cache is not read or written)' % (
        SWIFT['repo'], SWIFT['revision'], KEY, KEY)
    BOUND = '[QWEN-C2] checkpoint bound %s@%s key=%s snapshot=/s weight_cache=/experiment-cache/weights-ckpt/%s' % (SWIFT['repo'], SWIFT['revision'], KEY, KEY)

    def problems(self, text, expected=(SWIFT['repo'], SWIFT['revision'])):
        return checkpoint.container_log_problems(text, expected)

    def test_a_log_with_both_lines_is_clean(self):
        self.assertEqual(self.problems('\n'.join(['x', self.HEAD, 'weight cache /d: created', self.BOUND])), [])

    def test_each_missing_piece_is_named(self):
        self.assertTrue(any('never names the asked checkpoint' in p for p in self.problems(self.BOUND)))
        self.assertTrue(any('never logged' in p for p in self.problems(self.HEAD)))
        self.assertTrue(any('keyed weight cache' in p for p in self.problems(self.HEAD.replace('-ckpt/', '/') + '\n' + self.BOUND)))
        self.assertTrue(any('another selection' in p for p in self.problems(self.HEAD + '\n' + self.BOUND + '\n' + self.HEAD.replace(SWIFT['repo'], 'org/x'))))

    def test_no_checkpoint_asked_means_no_checkpoint_line(self):
        self.assertEqual(checkpoint.container_log_problems('[QWEN-C2] profile x: vLLM argv', None), [])
        self.assertTrue(checkpoint.container_log_problems(self.HEAD, None))

    def test_the_smoke_checks_flag_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke, container = Path(tmp) / 's.log', Path(tmp) / 'c.log'
            smoke.write_text('SMOKE_JSON {}\n')
            container.write_text(self.HEAD + '\n' + self.BOUND + '\n')
            with mock.patch.object(smoke_check, 'check', return_value=([], {})), mock.patch.object(smoke_check, 'slide_on', return_value=False), \
                    mock.patch.object(smoke_check, 'profile_env', return_value={}), mock.patch.object(smoke_check, 'profile_entry', return_value={}):
                self.assertEqual(smoke_check.main(['--smoke-log', str(smoke), '--container-log', str(container), '--profile', 'c2-packed-tp4',
                                                   '--expect-checkpoint', '%s@%s' % (SWIFT['repo'], SWIFT['revision'])]), 0)
                container.write_text('nothing\n')
                self.assertEqual(smoke_check.main(['--smoke-log', str(smoke), '--container-log', str(container), '--profile', 'c2-packed-tp4',
                                                   '--expect-checkpoint', '%s@%s' % (SWIFT['repo'], SWIFT['revision'])]), 1)
                self.assertEqual(smoke_check.main(['--smoke-log', str(smoke), '--container-log', str(container), '--profile', 'c2-packed-tp4']), 0, 'default off: not asked, not read')


class JobKeyTests(unittest.TestCase):
    PROFILE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4'

    def read(self, **values):
        base = dict(C2_IMAGE_TAG='tp4-swift-1', C2_CARDS='quad', C2_ACTIONS='reset smoke', C2_PROFILE=self.PROFILE, C2_SMOKE_TESTS='warmup,concurrent8_steady')
        base.update(values)
        profiles = job.profile_names(str(PROFILES))
        return job.read_job(base, profiles, envs=job.profile_envs(str(PROFILES)))

    def test_default_off_every_output_empty(self):
        outputs = self.read()
        self.assertEqual((outputs['model_repo'], outputs['model_revision'], outputs['model_served']), ('', '', ''))

    def test_swift_is_read_and_served_under_its_own_name(self):
        outputs = self.read(C2_MODEL_REPO=SWIFT['repo'], C2_MODEL_REVISION=SWIFT['revision'])
        self.assertEqual((outputs['model_repo'], outputs['model_revision'], outputs['model_served']), (SWIFT['repo'], SWIFT['revision'], 'ukisai/Swift-1.5-Qwen3.8-27b'))
        taulab = self.read(C2_MODEL_REPO=SWIFT['repo'], C2_MODEL_REVISION=SWIFT['revision'], C2_ACTIONS='reset taulab', C2_SMOKE_TESTS='')
        self.assertEqual(taulab['model_repo'], SWIFT['repo'])

    def test_every_misuse_is_refused(self):
        pair = dict(C2_MODEL_REPO=SWIFT['repo'], C2_MODEL_REVISION=SWIFT['revision'])
        for label, values in (('one alone', dict(C2_MODEL_REPO=SWIFT['repo'])), ('the other alone', dict(C2_MODEL_REVISION=SWIFT['revision'])),
                              ('a branch', dict(C2_MODEL_REPO=SWIFT['repo'], C2_MODEL_REVISION='main')), ('unlisted', dict(C2_MODEL_REPO='org/x', C2_MODEL_REVISION=SWIFT['revision'])),
                              ('the base', dict(C2_MODEL_REPO=BASE['repo'], C2_MODEL_REVISION=BASE['revision'])),
                              ('a pair job', dict(pair, C2_CARDS='pair', C2_PROFILE='general')),
                              ('a gate', dict(pair, C2_ACTIONS='reset gate')),
                              ('a prefix gate', dict(pair, C2_ACTIONS='reset prefix', C2_PREFIX_PLAN='tier-concurrent', C2_PREFIX_PROFILE=self.PROFILE, C2_PREFIX_BASELINE='none')),
                              ('a replay', dict(pair, C2_ACTIONS='reset replay', C2_PLATFORM_IMAGE='local:thin-layer', C2_REPLAY_PROFILE=self.PROFILE)),
                              ('a build', dict(pair, C2_ACTIONS='build')),
                              ('nothing to serve', dict(pair, C2_ACTIONS='status reset'))):
            with self.subTest(label), self.assertRaises(job.JobError) as raised:
                self.read(**values)
            self.assertIn('C2_MODEL', str(raised.exception))


class RequestDefaultTests(unittest.TestCase):
    """Swift's answer room: a request that omits max_tokens gets 16,384 (the output budget) instead of the profile's 8,192; the base keeps the profile's. The clamp to the room a long prompt
    leaves, the 16,384 ceiling and the per-request KV admission are the contract's own and unchanged."""
    SHIP = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']['c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4']

    def limits(self, environ):
        base = contract.request_limits(self.SHIP)
        with mock.patch.object(contract, 'log', lambda *args: None):
            return base, contract.checkpoint_limits(base, environ)

    def enforce(self, limits, prompt_tokens, max_tokens=None):
        params = type('Params', (), dict(n=1, logprobs=None, prompt_logprobs=None, structured_outputs=None, logit_bias=None, allowed_token_ids=None, bad_words=None, stop=None,
                                         min_tokens=0, stop_token_ids=None, max_tokens=max_tokens))()
        contract.enforce_request(params, prompt_tokens=prompt_tokens, max_model_len=262144, budget=limits['budget'], eos_ids=frozenset((248046, 248044)), max_prompt_tokens=limits['max_prompt_tokens'],
                                 min_answer_tokens=limits['min_answer_tokens'], default_max_tokens=limits['default_max_tokens'], drafter_headroom_tokens=limits.get('drafter_headroom_tokens', 0),
                                 kv_pool_blocks=limits.get('kv_pool_blocks'))
        return params.max_tokens

    def test_the_catalog_gives_swift_the_budget_and_the_base_nothing(self):
        self.assertEqual(SWIFT['request'], {'default_max_tokens': 16384})
        self.assertNotIn('request', BASE)
        self.assertEqual(checkpoint.request_overrides(checkpoint.selection(PAIR)), {'default_max_tokens': 16384})
        self.assertEqual(checkpoint.request_overrides(checkpoint.selection(BASE_PAIR)), {})
        self.assertEqual(checkpoint.request_overrides(None), {})

    def test_the_ship_profile_is_untouched_and_only_swift_changes_the_limit(self):
        base, off = self.limits({})
        self.assertIs(off, base, 'no selection: the very same limits, nothing read')
        self.assertEqual((base['default_max_tokens'], base['budget'], base['min_answer_tokens'], base['kv_pool_blocks']), (8192, 16384, 8192, 16499))
        self.assertEqual(self.limits(dict(BASE_PAIR))[1], base, 'the base pair: the profile\'s numbers')
        _, swift = self.limits(dict(PAIR))
        self.assertEqual(swift['default_max_tokens'], 16384)
        self.assertEqual({key: value for key, value in swift.items() if key != 'default_max_tokens'}, {key: value for key, value in base.items() if key != 'default_max_tokens'})

    def test_an_override_over_the_budget_is_refused(self):
        base = contract.request_limits(self.SHIP)
        with mock.patch.object(checkpoint, 'request_overrides', return_value={'default_max_tokens': 16385}), mock.patch.object(contract, 'log', lambda *args: None):
            with self.assertRaises(ValueError):
                contract.checkpoint_limits(base, dict(PAIR))

    def test_a_catalog_request_block_is_checked(self):
        for request in ({'default_max_tokens': 0}, {'default_max_tokens': '16384'}, {'other': 5}, [1]):
            with self.subTest(request=request), tempfile.TemporaryDirectory() as tmp:
                broken = copy.deepcopy(CATALOG)
                broken['checkpoints'][0]['request'] = request
                path = Path(tmp) / 'c.json'
                path.write_text(json.dumps(broken), encoding='utf-8')
                with self.assertRaises(checkpoint.CheckpointError):
                    checkpoint.load_catalog(str(path))
        broken = copy.deepcopy(CATALOG)
        broken['base']['request'] = {'default_max_tokens': 16384}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.json'
            path.write_text(json.dumps(broken), encoding='utf-8')
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.load_catalog(str(path))

    def test_the_default_is_clamped_to_the_room_a_long_prompt_leaves_and_an_explicit_value_is_the_budget_at_most(self):
        _, swift = self.limits(dict(PAIR))
        base = contract.request_limits(self.SHIP)
        self.assertEqual(self.enforce(swift, 1000), 16384)
        self.assertEqual(self.enforce(base, 1000), 8192, 'the base keeps its 8,192')
        self.assertEqual(self.enforce(swift, 200000), 16384)
        self.assertEqual(self.enforce(swift, 253920), 8192, 'the longest admitted prompt: 262,144 less the 32 headroom less 253,920 leaves 8,192')
        self.assertEqual(self.enforce(swift, 250000), 12112)
        self.assertEqual(self.enforce(swift, 1000, 20000), 16384, 'the ceiling is the budget: an explicit larger value is clamped, as before')
        self.assertEqual(self.enforce(swift, 1000, 5000), 5000)
        with self.assertRaises(contract.ContractError):
            self.enforce(swift, 253921)

    def test_every_default_request_is_admitted_by_the_kv_check_and_eight_fit_to_a_long_prompt(self):
        from serving_kv_reservation import request_blocks
        _, swift = self.limits(dict(PAIR))
        pool = swift['kv_pool_blocks']
        self.assertEqual(pool, 16499)
        for prompt in (1, 4096, 32768, 120000, 200000, 253920):
            granted = self.enforce(swift, prompt)
            self.assertLessEqual(request_blocks(prompt, granted), pool, 'a single default request is never refused for its blocks (%d tokens)' % prompt)
        # eight default requests at once, each holding its worst case: the longest prompt for which all eight fit in the pool
        def eight_fit(prompt, default):
            return 8 * request_blocks(prompt, min(default, 262144 - 32 - prompt)) <= pool

        for prompt in (4096, 32768, 65536, 100000, 115000):
            self.assertTrue(eight_fit(prompt, 16384), prompt)
        self.assertEqual(max(prompt for prompt in range(110000, 125000) if eight_fit(prompt, 16384)), 115488)
        self.assertEqual(max(prompt for prompt in range(110000, 130000) if eight_fit(prompt, 8192)), 123680)
        self.assertFalse(eight_fit(120000, 16384), 'between 115,488 and 123,680 prompt tokens the eighth DEFAULT-max_tokens request waits for blocks (the reservation holds it) where the 8,192 default ran it')


class PlatformModelTests(unittest.TestCase):
    """The model id the platform asks for must be the model the image serves: Swift is never answered under the base's name, nor the base under Swift's."""

    def test_the_catalog_rule(self):
        swift, base = checkpoint.selection(PAIR), checkpoint.selection(BASE_PAIR)
        for asked, sel, clean in ((BASE['repo'], None, True), (BASE['repo'], base, True), (SWIFT['repo'], swift, True), (None, swift, True), (None, None, True),
                                  ('/models/some/path', swift, True), ('org/unlisted', swift, True), ('org/unlisted', None, True),
                                  (SWIFT['repo'], None, False), (SWIFT['repo'], base, False), (BASE['repo'], swift, False),
                                  ('/models/models--ukisai--Swift-1.5-Qwen3.8-27b/snapshots/' + SWIFT['revision'], None, False),
                                  ('/models/models--Qwen--Qwen3.8-27B/snapshots/' + BASE['revision'], swift, False),
                                  ('/models/models--ukisai--Swift-1.5-Qwen3.8-27b/snapshots/' + SWIFT['revision'], swift, True)):
            with self.subTest(asked=asked, sel=sel):
                self.assertEqual(checkpoint.platform_model_problems(asked, sel, CATALOG) == [], clean)
        self.assertEqual(checkpoint.repo_of_path('/m/models--a--b-c/snapshots/' + 'f' * 40), 'a/b-c')
        self.assertIsNone(checkpoint.repo_of_path('/m/other'))

    def test_the_contract_reads_the_argv(self):
        self.assertEqual(contract.platform_model(['--port', '1', '--model', 'x/y']), 'x/y')
        self.assertEqual(contract.platform_model(['--model=x/y']), 'x/y')
        self.assertIsNone(contract.platform_model(['--port', '1']))
        self.assertIsNone(contract.platform_model(['--model']))

    def test_the_production_path_is_free_and_a_mismatch_refuses_the_boot(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop('qwen_c2_checkpoint', None)
            contract.check_platform_model(['--model', 'Qwen/Qwen3.8-27B'], {})
            contract.check_platform_model([], {})
            self.assertNotIn('qwen_c2_checkpoint', sys.modules, 'the base id on an unselected image costs no import')
        contract.check_platform_model(['--model', SWIFT['repo']], dict(PAIR))
        for argv, environ in ((['--model', SWIFT['repo']], {}), (['--model', BASE['repo']], dict(PAIR)), (['--model=' + SWIFT['repo']], dict(BASE_PAIR))):
            with self.subTest(argv=argv, environ=environ), self.assertRaises(ValueError) as raised:
                contract.check_platform_model(argv, environ)
            self.assertIn('the platform asks for the model', str(raised.exception))


class PlumbingTests(unittest.TestCase):
    """The pair rides the docker argv of the steps that serve (agent_shape, server_run), the tau lab's launched-argv check and the workflow's smoke step; off, nothing moves."""

    def setUp(self):
        import argparse
        import c2_prefix_gate as prefix_gate
        import c2_serving_gate as gate
        import c2_tau_lab as lab
        self.argparse, self.prefix_gate, self.gate, self.lab = argparse, prefix_gate, gate, lab

    def test_agent_shape_adds_exactly_the_two_variables_and_only_when_asked(self):
        plain = self.gate.agent_shape('img', 'n', 'prof', ['/dev/a'])
        self.assertEqual(self.gate.agent_shape('img', 'n', 'prof', ['/dev/a'], model=None), plain)
        asked = self.gate.agent_shape('img', 'n', 'prof', ['/dev/a'], model=(SWIFT['repo'], SWIFT['revision']))
        self.assertEqual(asked, plain + ['-e', 'QWEN_C2_MODEL_REPO=' + SWIFT['repo'], '-e', 'QWEN_C2_MODEL_REVISION=' + SWIFT['revision']])

    def test_the_prefix_gates_server_run_passes_it_through(self):
        plain = self.prefix_gate.server_run('img', 'n', 'prof', ['/dev/a'])
        self.assertEqual(self.prefix_gate.server_run('img', 'n', 'prof', ['/dev/a'], model=None), plain)
        asked = self.prefix_gate.server_run('img', 'n', 'prof', ['/dev/a'], model=(SWIFT['repo'], SWIFT['revision']))
        self.assertEqual(len(asked), len(plain) + 4)
        self.assertIn('QWEN_C2_MODEL_REVISION=' + SWIFT['revision'], asked)
        self.assertEqual(asked[asked.index('--model') + 1], SWIFT['repo'], 'the platform asks for the checkpoint it is served')
        self.assertEqual(plain[plain.index('--model') + 1], 'Qwen/Qwen3.8-27B')
        at = asked.index('QWEN_C2_MODEL_REPO=' + SWIFT['repo'])
        stripped = asked[:at - 1] + asked[at + 3:]
        stripped[stripped.index('--model') + 1] = 'Qwen/Qwen3.8-27B'
        self.assertEqual(stripped, plain, 'removing the two -e pairs and the model id gives the argv it was')

    def test_the_tau_lab_reads_its_flags_and_refuses_what_the_job_reader_refuses(self):
        def options(repo, revision):
            return self.argparse.Namespace(model_repo=repo, model_revision=revision)

        self.assertIsNone(self.lab.read_model(options(None, None)))
        self.assertEqual(self.lab.read_model(options(SWIFT['repo'], SWIFT['revision'])), (SWIFT['repo'], SWIFT['revision']))
        for repo, revision in ((SWIFT['repo'], None), (None, SWIFT['revision']), (SWIFT['repo'], 'main'), (BASE['repo'], BASE['revision']), ('org/x', SWIFT['revision'])):
            with self.subTest(repo=repo, revision=revision), self.assertRaises(self.lab.LabError):
                self.lab.read_model(options(repo, revision))
        parsed = self.lab.build_parser().parse_args(['--data', 'd', '--results', 'r', '--model-repo', SWIFT['repo'], '--model-revision', SWIFT['revision']])
        self.assertEqual((parsed.model_repo, parsed.model_revision), (SWIFT['repo'], SWIFT['revision']))

    def test_the_tau_labs_launched_check_needs_the_checkpoint_lines_when_asked_and_forbids_them_when_not(self):
        derived = 'c2-packed-tp4-lab'
        base_lines = ['[QWEN-C2] profile %s: vLLM argv' % derived]
        self.assertEqual(self.lab.launched_problems(base_lines, derived), [])
        self.assertEqual(self.lab.launched_problems(base_lines, derived, None, None), [])
        asked = (SWIFT['repo'], SWIFT['revision'])
        self.assertTrue(any('never names the asked checkpoint' in problem for problem in self.lab.launched_problems(base_lines, derived, None, asked)))
        both = base_lines + [LogReadTests.HEAD, LogReadTests.BOUND]
        self.assertEqual(self.lab.launched_problems(both, derived, None, asked), [])
        self.assertTrue(self.lab.launched_problems(both, derived), 'a checkpoint line in a run that asked for none is a problem')

    def test_the_workflows_smoke_step_forwards_the_pair_and_keeps_the_default_argv(self):
        text = (HERE.parent.parent / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        start = text.index('- name: Smoke on the four-card set')
        step = text[start:text.index('- name: Replay the node agent', start)]
        for needle in ('MODEL_REPO: ${{ steps.job.outputs.model_repo }}', 'MODEL_REVISION: ${{ steps.job.outputs.model_revision }}', 'SERVED_MODEL: ${{ steps.job.outputs.model_served }}',
                       'served="${SERVED_MODEL:-Qwen/Qwen3.8-27B}"', '--model "${MODEL_REPO:-Qwen/Qwen3.8-27B}" --served-model-name "$served"', '${MODEL_REPO:+-e "QWEN_C2_MODEL_REPO=$MODEL_REPO" -e "QWEN_C2_MODEL_REVISION=$MODEL_REVISION"}',
                       'c2_serving_smoke.py http://127.0.0.1:8010 "$served"', '${MODEL_REPO:+--expect-checkpoint "$MODEL_REPO@$MODEL_REVISION"}', 'CODING_SET_OUT="$results/coding-set-$PROFILE.json"'):
            self.assertIn(needle, step)
        lab_step = text[text.index('- name: Tau lab (W-T1) on the four-card set'):]
        self.assertIn('args+=(--model-repo "$MODEL_REPO" --model-revision "$MODEL_REVISION")', lab_step)


if __name__ == '__main__':
    unittest.main()
