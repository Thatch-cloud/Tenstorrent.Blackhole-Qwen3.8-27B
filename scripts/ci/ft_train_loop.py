"""WP-F2: the drafter fine-tune's training loop on CPU, the golden reference for the ttml port: AdamW at a constant 5e-4 after a 5%
warmup, gradient clip 1.0, the frozen fc / hidden_norm / embedding / LM head, checkpoint and EXACT resume, and export to the checkpoint's
tensor names (the `dedf8df6` layout) with a round-trip check.

`Trainer.step(example)` runs one optimizer step on one example (input_ids [1, S], loss_mask [1, S], raw tap rows [1, S, taps * H]);
`accumulate` averages the gradients of several examples first (the gradient-accumulation and DDP reference: DDP averages the
per-rank losses, and every rank normalises by its own weight sum, so the reference averages per-example losses, not one pooled loss).
A checkpoint carries the model, the optimizer, the step and the generator state: resuming after N steps reproduces an uninterrupted run
bit for bit (CPU, fp32).
"""
import torch

import ft_dflash2_train as train
import ft_loss

LR = 5e-4
WARMUP_RATIO = 0.05
CLIP = 1.0


def lr_at(step, total_steps, lr=LR, warmup_ratio=WARMUP_RATIO):
    """Linear warmup over the first warmup_ratio of the steps, then constant. `step` counts from 0."""
    warmup = max(1, int(round(total_steps * warmup_ratio)))
    return lr * min(1.0, (step + 1) / float(warmup))


class Trainer(object):
    def __init__(self, model, embed_weight, lm_head_weight, total_steps, num_anchors=512, lr=LR, warmup_ratio=WARMUP_RATIO, clip=CLIP,
                 gamma=ft_loss.GAMMA, alpha=ft_loss.ALPHA, block_rule='specforge', frozen=train.FROZEN_DEFAULT, seed=0, chunk_blocks=None,
                 weight_decay=0.0):
        self.model, self.embed, self.head = model, embed_weight, lm_head_weight
        self.total_steps, self.num_anchors, self.lr, self.warmup_ratio, self.clip = total_steps, num_anchors, lr, warmup_ratio, clip
        self.gamma, self.alpha, self.block_rule, self.chunk_blocks = gamma, alpha, block_rule, chunk_blocks
        self.frozen = train.freeze(model, frozen)
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        self.generator = torch.Generator().manual_seed(seed)
        self.step_count = 0

    def loss_of(self, example):
        input_ids, loss_mask, raw_rows = example
        loss, terms, _ = train.train_forward(self.model, self.embed, self.head, raw_rows, input_ids, loss_mask, self.num_anchors,
                                             self.generator, self.gamma, self.alpha, None, self.block_rule, self.chunk_blocks)
        return loss, terms

    def step(self, examples):
        """One optimizer step over `examples` (a list): the mean of their per-example losses. -> (loss value, grad norm)."""
        examples = list(examples)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for example in examples:
            loss, _ = self.loss_of(example)
            (loss / len(examples)).backward()
            total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.clip)
        for group in self.optimizer.param_groups:
            group['lr'] = lr_at(self.step_count, self.total_steps, self.lr, self.warmup_ratio)
        self.optimizer.step()
        self.step_count += 1
        return total / len(examples), float(norm)

    # -- checkpoint ---------------------------------------------------------------------------------------------------------
    def state(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), step=self.step_count,
                    generator=self.generator.get_state())

    def load(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.step_count = state['step']
        self.generator.set_state(state['generator'])

    def save(self, path):
        torch.save(self.state(), path)

    def resume(self, path):
        self.load(torch.load(path, weights_only=False))


# -- export to the checkpoint layout ------------------------------------------------------------------------------------------------

def export_state(model, dtype=torch.float32):
    """{checkpoint tensor name: tensor} on CPU in `dtype` (bf16 for the released checkpoint)."""
    return dict((name, value.detach().to('cpu').to(dtype).clone()) for name, value in model.state_dict().items())


def import_state(model, tensors):
    """Load an exported dict into a model; the names and shapes must match exactly."""
    own = model.state_dict()
    if set(own) != set(tensors):
        raise ValueError('exported names differ from the model: %d missing, %d extra' % (len(set(own) - set(tensors)),
                                                                                         len(set(tensors) - set(own))))
    for name, value in tensors.items():
        if tuple(own[name].shape) != tuple(value.shape):
            raise ValueError('shape of %s differs' % name)
    model.load_state_dict(dict((name, value.to(own[name].dtype)) for name, value in tensors.items()))
    return model


def max_abs_difference(a, b):
    """The largest element-wise difference over two exported dicts of the same names (the compare step of train-export-compare)."""
    if set(a) != set(b):
        raise ValueError('different tensor names')
    return max((float((a[name].float() - b[name].float()).abs().max()) for name in a), default=0.0)
