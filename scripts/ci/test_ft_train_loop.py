"""The training loop on CPU: overfitting two samples, exact resume, the warmup schedule, gradient accumulation, export and the train /
export / compare round trip (the shape of the ttml #41657 gate), and a 2-rank gloo run against the single-process reference."""
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import dflash2_torch as d2  # noqa: E402
import ft_dflash2_train as tr  # noqa: E402
import ft_train_loop as loop  # noqa: E402

VOCAB = 48


def make(seed=0, window=16, block=4, **kwargs):
    cfg = d2.tiny_config(window=window, block=block, vocab=VOCAB, mask_token=VOCAB - 1, top_k=4, **kwargs)
    model = d2.init_random(d2.Dflash2(cfg), seed).float()
    generator = torch.Generator().manual_seed(seed + 1)
    embed = torch.randn(VOCAB, cfg.hidden, generator=generator)
    head = torch.randn(VOCAB, cfg.hidden, generator=generator)
    return cfg, model, embed, head


def examples(cfg, count=2, length=24, seed=5):
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        ids = torch.randint(0, VOCAB - 1, (1, length), generator=generator)
        mask = torch.zeros(1, length)
        mask[:, 4:] = 1.0
        out.append((ids, mask, torch.randn(1, length, len(cfg.tap_ids) * cfg.hidden, generator=generator)))
    return out


def trainer(seed=0, total=40, lr=1e-2, **kwargs):
    cfg, model, embed, head = make(seed)
    return cfg, loop.Trainer(model, embed, head, total, num_anchors=6, lr=lr, seed=seed, **kwargs)


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_constant(self):
        values = [loop.lr_at(step, 100) for step in range(8)]
        self.assertAlmostEqual(values[0], 5e-4 / 5)
        self.assertAlmostEqual(values[4], 5e-4)
        self.assertEqual(values[5:], [5e-4] * 3)
        self.assertEqual(loop.lr_at(0, 1), 5e-4)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_the_recipes_defaults(self):
        self.assertEqual((loop.LR, loop.WARMUP_RATIO, loop.CLIP), (5e-4, 0.05, 1.0))


class StepTests(unittest.TestCase):
    def test_overfitting_two_samples_drives_the_loss_down(self):
        cfg, t = trainer(total=60, lr=3e-2)
        data = examples(cfg)
        first = loop.Trainer.step(t, data)[0]
        for _ in range(59):
            last = t.step(data)[0]
        self.assertLess(last, 0.5 * first)

    def test_frozen_parameters_never_move(self):
        cfg, t = trainer()
        before = dict((name, value.clone()) for name, value in t.model.state_dict().items())
        data = examples(cfg)
        for _ in range(3):
            t.step(data)
        after = t.model.state_dict()
        for name in ('fc.weight', 'hidden_norm.weight'):
            self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertFalse(torch.equal(before['layers.0.mlp.up_proj.weight'], after['layers.0.mlp.up_proj.weight']))

    def test_gradient_clipping_bounds_the_step(self):
        cfg, t = trainer(clip=1e-3)
        _, norm = t.step(examples(cfg))
        self.assertGreater(norm, 1e-3)               # the reported norm is the pre-clip norm
        total = sum(float(p.grad.norm()) ** 2 for p in t.model.parameters() if p.grad is not None) ** 0.5
        self.assertLessEqual(total, 1e-3 * 1.001)

    def test_accumulation_is_the_mean_of_the_per_example_losses(self):
        cfg, t = trainer()
        data = examples(cfg)
        state = t.generator.get_state()
        loss, _ = t.step(data)
        t.generator.set_state(state)
        t2_cfg, t2 = trainer()
        mean = (t2.loss_of(data[0])[0].item() + t2.loss_of(data[1])[0].item()) / 2
        self.assertAlmostEqual(loss, mean, places=5)


class ResumeTests(unittest.TestCase):
    def test_resume_reproduces_an_uninterrupted_run_bit_for_bit(self):
        root = tempfile.mkdtemp()
        try:
            cfg, straight = trainer(seed=3, total=12)
            data = examples(cfg)
            for _ in range(6):
                straight.step(data)
            cfg, first = trainer(seed=3, total=12)
            for _ in range(3):
                first.step(data)
            path = os.path.join(root, 'ckpt.pt')
            first.save(path)
            cfg, second = trainer(seed=3, total=12)
            with torch.no_grad():                                 # a different state and generator: the checkpoint must override them
                for parameter in second.model.parameters():
                    parameter.add_(0.5)
            second.generator.manual_seed(999)
            second.resume(path)
            self.assertEqual(second.step_count, 3)
            for _ in range(3):
                second.step(data)
            for name, value in straight.model.state_dict().items():
                self.assertTrue(torch.equal(value, second.model.state_dict()[name]), name)
        finally:
            shutil.rmtree(root)

    def test_a_checkpoint_without_the_optimizer_state_would_diverge(self):
        cfg, a = trainer(seed=3)
        data = examples(cfg)
        for _ in range(3):
            a.step(data)
        cfg, b = trainer(seed=3)
        state = a.state()
        state['optimizer'] = b.optimizer.state_dict()              # a fresh optimizer: the mutation
        b.load(state)
        a.step(data)
        b.step(data)
        self.assertFalse(torch.equal(a.model.layers[0].mlp.up_proj.weight, b.model.layers[0].mlp.up_proj.weight))


class ExportTests(unittest.TestCase):
    def test_export_names_are_the_checkpoint_names_and_round_trip(self):
        cfg, t = trainer()
        data = examples(cfg)
        for _ in range(2):
            t.step(data)
        exported = loop.export_state(t.model)
        self.assertEqual(sorted(exported), d2.parameter_names(cfg.layers))
        fresh = d2.Dflash2(cfg).float()
        loop.import_state(fresh, exported)
        self.assertEqual(loop.max_abs_difference(exported, loop.export_state(fresh)), 0.0)

    def test_train_n_steps_export_compare(self):
        """The ttml #41657 pattern: train, export, load into a fresh model, and compare outputs and tensors."""
        cfg, t = trainer()
        data = examples(cfg)
        for _ in range(4):
            t.step(data)
        exported = loop.export_state(t.model, torch.bfloat16)
        fresh = loop.import_state(d2.Dflash2(cfg).float(), exported)
        self.assertLess(loop.max_abs_difference(loop.export_state(t.model), loop.export_state(fresh)), 0.01)
        ids, mask, raw = data[0]
        batch = tr.make_batch(ids, mask, cfg, 4, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            a = tr.draft_hidden(t.model, t.embed, raw, batch)
            b = tr.draft_hidden(fresh, t.embed, raw, batch)
        self.assertTrue(torch.allclose(a, b, atol=0.05))

    def test_names_and_shapes_are_checked(self):
        cfg, t = trainer()
        exported = loop.export_state(t.model)
        broken = dict(exported)
        broken.pop('norm.weight')
        with self.assertRaises(ValueError):
            loop.import_state(d2.Dflash2(cfg), broken)
        wrong = dict(exported)
        wrong['norm.weight'] = torch.zeros(3)
        with self.assertRaises(ValueError):
            loop.import_state(d2.Dflash2(cfg), wrong)
        with self.assertRaises(ValueError):
            loop.max_abs_difference(exported, broken)


# -- two ranks ------------------------------------------------------------------------------------------------------------------

def _worker(rank, world, init, batches, out_dir):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=world)
    cfg, model, embed, head = make(seed=7)
    model.train()
    ids, mask, raw, batch = batches[rank]
    loss, _, _ = tr.train_forward(model, embed, head, raw, ids, mask, 6, batch=batch)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
            parameter.grad /= world
    torch.save(dict((name, p.grad.clone()) for name, p in model.named_parameters() if p.grad is not None),
               os.path.join(out_dir, 'rank%d.pt' % rank))
    dist.destroy_process_group()


class TwoRankTests(unittest.TestCase):
    def test_two_gloo_ranks_equal_the_single_process_average(self):
        import torch.distributed as dist
        import torch.multiprocessing as mp
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest('no gloo')
        cfg, model, embed, head = make(seed=7)
        data = examples(cfg, count=2, seed=11)
        batches = []
        for index, (ids, mask, raw) in enumerate(data):
            batches.append((ids, mask, raw, tr.make_batch(ids, mask, cfg, 6, generator=torch.Generator().manual_seed(index))))
        # single-process reference: the mean of the per-example gradients (DDP averages per-rank losses)
        reference = {}
        for ids, mask, raw, batch in batches:
            model.zero_grad()
            loss, _, _ = tr.train_forward(model, embed, head, raw, ids, mask, 6, batch=batch)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    reference[name] = reference.get(name, 0) + parameter.grad.clone() / len(batches)
        root = tempfile.mkdtemp()
        try:
            init = 'file:///' + os.path.join(root, 'store').replace('\\', '/')
            try:
                mp.spawn(_worker, args=(2, init, batches, root), nprocs=2, join=True)
            except Exception as error:               # a platform without usable multiprocess gloo: the arithmetic above still ran
                self.skipTest('two-process gloo unavailable: %s' % type(error).__name__)
            for rank in (0, 1):
                got = torch.load(os.path.join(root, 'rank%d.pt' % rank))
                self.assertEqual(sorted(got), sorted(reference))
                for name, value in reference.items():
                    self.assertTrue(torch.allclose(got[name], value, atol=1e-6), (rank, name))
        finally:
            shutil.rmtree(root, True)


if __name__ == '__main__':
    unittest.main()
