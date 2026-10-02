"""a0_drafters at tiny scale: the Markov chain against the CPU reference, the walkers against a from-scratch recompute, windows,
the context cache, the look-ahead guard, the upstream adapter's bookkeeping."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import a0_drafters as dr  # noqa: E402
import dflash2_torch as d2  # noqa: E402
import dspark_markov  # noqa: E402
import tf_pair_walk as walk  # noqa: E402

VOCAB = 64


class Features(object):
    """rows(a, b) over a fixed tensor; records the highest row asked for."""
    def __init__(self, tensor):
        self.tensor, self.highest = tensor, 0

    def rows(self, a, b):
        self.highest = max(self.highest, b)
        return self.tensor[a:b]


def setup(window=16, seed=0, vocab=VOCAB):
    cfg = d2.tiny_config(window=window, vocab=vocab, mask_token=vocab - 1, top_k=4)
    model = d2.init_random(d2.Dflash2(cfg), seed).float().eval()
    generator = torch.Generator().manual_seed(seed + 1)
    embed = torch.randn(vocab, cfg.hidden, generator=generator)
    head = torch.randn(vocab, cfg.hidden, generator=generator)
    tensor = torch.randn(300, len(cfg.tap_ids) * cfg.hidden, generator=generator)
    return cfg, model, embed, head, tensor


def direct_dflash2(cfg, model, embed, head, tensor, sequence, start, count, window):
    """From scratch: no cache, the whole context slice at once."""
    lo = max(0, start - window) if window is not None else 0
    noise = embed[torch.tensor([sequence[start]] + [cfg.mask_token] * count)][None]
    positions = (start + torch.arange(count + 1))[None]
    with torch.no_grad():
        hidden = model(noise, positions, tensor[lo:start][None], (lo + torch.arange(start - lo))[None])
        return model.propose(hidden[:, 1:], torch.tensor([sequence[start]]), head)[0].tolist()


class MarkovTests(unittest.TestCase):
    def test_chain_equals_the_cpu_reference(self):
        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(1, 6, 40, generator=generator)
        pred, succ = torch.randn(40, 3, generator=generator), torch.randn(40, 3, generator=generator)
        reference = dspark_markov.greedy_proposals(logits, torch.tensor([7]), pred, succ)
        mine = dr.markov_chain(logits[0], 7, pred, succ)
        self.assertEqual(mine.tolist(), reference[0].tolist())

    def test_the_bias_matters(self):
        generator = torch.Generator().manual_seed(6)
        logits = torch.randn(5, 40, generator=generator)
        zero = torch.zeros(40, 3)
        plain = dr.markov_chain(logits, 7, zero, zero)
        self.assertEqual(plain.tolist(), logits.argmax(-1).tolist())
        pred, succ = 3 * torch.randn(40, 3, generator=generator), 3 * torch.randn(40, 3, generator=generator)
        self.assertNotEqual(dr.markov_chain(logits, 7, pred, succ).tolist(), plain.tolist())


class Dflash2WalkerTests(unittest.TestCase):
    def walker(self, window=16, count=7, seed=0):
        cfg, model, embed, head, tensor = setup(window, seed)
        backbone = dr.PortBackbone(model, embed)
        walker = dr.Dflash2Walker(backbone, head, model, cfg.mask_token, count, window)
        return cfg, model, embed, head, tensor, backbone, walker

    def test_cached_walk_equals_a_recompute_at_every_start(self):
        cfg, model, embed, head, tensor, backbone, walker = self.walker()
        sequence = [int(x) for x in torch.randint(0, VOCAB - 1, (300,))]
        walker.begin(Features(tensor))
        # forward steps of 1..5 rows, then a jump past the window, then small steps again
        for start in (20, 21, 24, 29, 31, 80, 83, 84, 200):
            got = walker.propose(sequence, start, 7)
            want = direct_dflash2(cfg, model, embed, head, tensor, sequence, start, 7, 16)
            self.assertEqual(got, want, start)

    def test_the_context_cache_stays_bounded_by_the_window(self):
        cfg, model, embed, head, tensor, backbone, walker = self.walker(window=16)
        sequence = [1] * 300
        walker.begin(Features(tensor))
        for start in range(20, 250, 7):
            walker.propose(sequence, start, 7)
            self.assertLessEqual(backbone.high - backbone.low, 16 + 1)
            self.assertEqual(backbone.high, start)

    def test_only_rows_below_the_anchor_are_read(self):
        cfg, model, embed, head, tensor, backbone, walker = self.walker()
        features = Features(tensor)
        walker.begin(features)
        walker.propose([1] * 300, 100, 7)
        self.assertLessEqual(features.highest, 100)

    def test_a_new_turn_starts_clean(self):
        cfg, model, embed, head, tensor, backbone, walker = self.walker()
        sequence = [int(x) for x in torch.randint(0, VOCAB - 1, (300,))]
        walker.begin(Features(tensor))
        walker.propose(sequence, 150, 7)
        walker.begin(Features(tensor * 2))
        got = walker.propose(sequence, 40, 7)
        self.assertEqual(got, direct_dflash2(cfg, model, embed, head, tensor * 2, sequence, 40, 7, 16))

    def test_wrong_count_refused(self):
        cfg, model, embed, head, tensor, backbone, walker = self.walker()
        walker.begin(Features(tensor))
        with self.assertRaises(ValueError):
            walker.propose([1] * 300, 50, 15)

    def test_t16_and_t8_blocks(self):
        for count in (15, 7):
            cfg, model, embed, head, tensor, backbone, walker = self.walker(count=count)
            walker.begin(Features(tensor))
            self.assertEqual(len(walker.propose([2] * 300, 60, count)), count)

    def test_end_to_end_with_the_walk(self):
        cfg, model, embed, head, tensor, backbone, walker = self.walker(count=7)
        generator = torch.Generator().manual_seed(9)
        turn = dict(prompt_ids=[int(x) for x in torch.randint(0, VOCAB - 1, (120,), generator=generator)],
                    output_ids=[int(x) for x in torch.randint(0, VOCAB - 1, (60,), generator=generator)])
        walker.begin(Features(tensor))
        rounds = walk.walk_free(turn, walker, 7)
        self.assertTrue(rounds)
        self.assertTrue(all(entry['committed'] >= 1 for entry in rounds))
        # a drafter reading the logged answer would see an IndexError from the walk's view: this one does not read it
        self.assertEqual(rounds[0]['start'], 120)


class DSparkWalkerTests(unittest.TestCase):
    def build(self, window=None, count=7, seed=1):
        cfg, model, embed, head, tensor = setup(window=None, seed=seed)
        generator = torch.Generator().manual_seed(seed + 7)
        pred, succ = torch.randn(VOCAB, 4, generator=generator), torch.randn(VOCAB, 4, generator=generator)
        backbone = dr.PortBackbone(model, embed)
        return cfg, model, embed, head, tensor, pred, succ, dr.DSparkWalker(backbone, head, pred, succ, cfg.mask_token, count, window)

    def direct(self, cfg, model, embed, head, tensor, pred, succ, sequence, start, count, window):
        lo = max(0, start - window) if window is not None else 0
        noise = embed[torch.tensor([sequence[start]] + [cfg.mask_token] * (count - 1))][None]
        positions = (start + torch.arange(count))[None]
        with torch.no_grad():
            hidden = model(noise, positions, tensor[lo:start][None], (lo + torch.arange(start - lo))[None])
            return dspark_markov.greedy_proposals(hidden @ head.T, torch.tensor([sequence[start]]), pred, succ)[0].tolist()

    def test_full_attention_walk_equals_a_recompute(self):
        cfg, model, embed, head, tensor, pred, succ, walker = self.build()
        sequence = [int(x) for x in torch.randint(0, VOCAB - 1, (300,))]
        walker.begin(Features(tensor))
        for start in (10, 11, 17, 90, 91, 250):
            got = walker.propose(sequence, start, 7)
            self.assertEqual(got, self.direct(cfg, model, embed, head, tensor, pred, succ, sequence, start, 7, None), start)

    def test_window_arm_slices_the_context_exactly(self):
        for window in (8, 40):
            cfg, model, embed, head, tensor, pred, succ, walker = self.build(window=window)
            sequence = [int(x) for x in torch.randint(0, VOCAB - 1, (300,))]
            walker.begin(Features(tensor))
            for start in (30, 33, 100, 101, 250):
                got = walker.propose(sequence, start, 7)
                self.assertEqual(got, self.direct(cfg, model, embed, head, tensor, pred, succ, sequence, start, 7, window), (window, start))

    def test_the_window_changes_the_answer_somewhere(self):
        cfg, model, embed, head, tensor, pred, succ, full = self.build(window=None)
        _, _, _, _, _, _, _, narrow = self.build(window=4)
        sequence = [int(x) for x in torch.randint(0, VOCAB - 1, (300,))]
        full.begin(Features(tensor))
        narrow.begin(Features(tensor))
        differs = sum(full.propose(sequence, s, 7) != narrow.propose(sequence, s, 7) for s in range(40, 280, 6))
        self.assertGreater(differs, 0)

    def test_rows_are_count_with_the_anchor_first(self):
        cfg, model, embed, head, tensor, pred, succ, walker = self.build(count=15)
        walker.begin(Features(tensor))
        self.assertEqual(len(walker.propose([3] * 300, 60, 15)), 15)


class FakeCache(object):
    def __init__(self):
        self.length = 0


class FakeUpstream(object):
    """The upstream call shape: rows in, rows out, the cache grows by context + block rows."""
    def __init__(self):
        self.calls = []

    def __call__(self, position_ids, noise_embedding, target_hidden, past_key_values, use_cache):
        self.calls.append((position_ids[0, 0].item(), position_ids[0, -1].item(), target_hidden.shape[1], noise_embedding.shape[1]))
        past_key_values.length += target_hidden.shape[1] + noise_embedding.shape[1]
        return torch.zeros(1, noise_embedding.shape[1], 4)


class UpstreamBackboneTests(unittest.TestCase):
    def test_context_is_passed_once_and_the_cache_is_cropped_back(self):
        model = FakeUpstream()
        crops = []

        def crop(cache, length):
            crops.append(length)
            cache.length = length

        backbone = dr.UpstreamBackbone(model, torch.zeros(10, 4), FakeCache, crop)
        features = Features(torch.zeros(100, 6))
        backbone.block_hidden(features, [1, 2, 3], 30, 16)
        backbone.block_hidden(features, [1, 2, 3], 35, 16)
        backbone.block_hidden(features, [1, 2, 3], 80, 16)
        # first call: rows [14, 30); second: only the 5 new rows; third: the window moved past the cache -> a fresh cache, rows [64, 80)
        self.assertEqual(model.calls[0], (14, 32, 16, 3))
        self.assertEqual(model.calls[1], (30, 37, 5, 3))
        self.assertEqual(model.calls[2], (64, 82, 16, 3))
        self.assertEqual(crops, [16, 21, 16])
        self.assertLessEqual(features.highest, 80)


class WeightsAndV4Tests(unittest.TestCase):
    def test_load_port_state_round_trip_and_refusals(self):
        cfg, source, _, _, _ = setup()
        target = d2.Dflash2(cfg)
        dr.load_port_state(target, dict(source.state_dict()))
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[name], value))
        broken = dict(source.state_dict())
        broken.pop('fc.weight')
        with self.assertRaises(ValueError):
            dr.load_port_state(target, broken)
        extra = dict(source.state_dict())
        extra['confidence_head.proj.weight'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            dr.load_port_state(target, extra)
        dr.load_port_state(target, extra, ignore=('confidence_head.proj.weight',))
        wrong = dict(source.state_dict())
        wrong['fc.weight'] = torch.zeros(3, 3)
        with self.assertRaises(ValueError):
            dr.load_port_state(target, wrong)

    def test_v4_round_identical_devices_agree_everywhere(self):
        generator = torch.Generator().manual_seed(3)
        hidden = torch.randn(7, 16, generator=generator)
        head = torch.randn(VOCAB, 16, generator=generator)
        pred, succ = torch.randn(VOCAB, 4, generator=generator), torch.randn(VOCAB, 4, generator=generator)
        answer = dr.markov_chain(hidden @ head.T, 5, pred, succ).tolist()
        rows, (gpu, cpu) = dr.v4_round(hidden, hidden.clone(), head, pred, succ, 5, answer, lambda p, a: len([1 for x, y in zip(p, a) if x == y]))
        self.assertTrue(all(agree for agree, _ in rows))
        self.assertEqual((gpu, cpu), (7, 7))
        self.assertTrue(all(margin >= 0 for _, margin in rows))

    def test_v4_round_sees_a_diverging_device(self):
        generator = torch.Generator().manual_seed(4)
        hidden = torch.randn(7, 16, generator=generator)
        head = torch.randn(VOCAB, 16, generator=generator)
        pred, succ = torch.randn(VOCAB, 4, generator=generator), torch.randn(VOCAB, 4, generator=generator)
        answer = dr.markov_chain(hidden @ head.T, 5, pred, succ).tolist()
        rows, (gpu, cpu) = dr.v4_round(hidden, hidden + 3 * torch.randn(7, 16, generator=generator), head, pred, succ, 5, answer,
                                       lambda p, a: len([1 for x, y in zip(p, a) if x == y]))
        self.assertEqual(gpu, 7)
        self.assertFalse(all(agree for agree, _ in rows))


if __name__ == '__main__':
    unittest.main()
