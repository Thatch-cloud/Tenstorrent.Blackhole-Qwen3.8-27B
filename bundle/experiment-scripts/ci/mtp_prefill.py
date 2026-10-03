"""Shifted MTP prompt pairs and target-to-draft cache-position alignment."""


class AlignedMTPStep:
    def __init__(self, device_step):
        self.device_step = device_step

    def __call__(self, token, hidden, position, *, select):
        if type(position) is not int or position < 1:
            raise ValueError('MTP input token needs its preceding target hidden position')
        return self.device_step(token, hidden, position - 1, select=select)


def initialize_prompt(device_step, prompt, hidden_rows, anchor, *, stage_row, copy_hidden):
    import torch

    prompt = tuple(prompt)
    if (not prompt or len(prompt) > 256
            or any(type(token) is not int or not 0 <= token < 248320 for token in prompt)
            or hidden_rows.device.type != 'cpu' or hidden_rows.dtype != torch.bfloat16
            or hidden_rows.ndim != 4 or tuple(hidden_rows.shape[:2]) != (1, 1)
            or hidden_rows.shape[3] != 5120 or not len(prompt) <= hidden_rows.shape[2] <= 256
            or not torch.isfinite(hidden_rows[:, :, :len(prompt)]).all()
            or not all(callable(callback) for callback in (device_step, stage_row, copy_hidden))):
        raise ValueError('Valid short prompt and complete CPU BF16 normalized target prefill rows required')
    for position in range(len(prompt) - 1):
        hidden = stage_row(hidden_rows[:, :, position:position + 1])
        device_step(prompt[position + 1], hidden, position, select=False)
    copy_hidden(stage_row(hidden_rows[:, :, len(prompt) - 1:len(prompt)]), anchor)
    return dict(prompt_tokens=len(prompt), initialized_mtp_rows=len(prompt) - 1,
                next_mtp_position=len(prompt) - 1, target_position=len(prompt),
                target_to_mtp_position_offset=-1, excluded_padding=hidden_rows.shape[2] - len(prompt))
