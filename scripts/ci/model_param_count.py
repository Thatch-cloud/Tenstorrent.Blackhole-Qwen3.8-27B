"""Count the model's parameters and FLOPs, so prefill MFU stops being a guess.

The corrected prefill profile puts arithmetic at 39-43% of device time while
prefill measures 11.7x off compute-bound. Those two together say the matmuls are
slow when they run - but only if the FLOP requirement is right, and so far it has
been inferred from the model NAME (27B) rather than counted from the architecture.

That is exactly the move that produced the 9.3 GB/s error: a derived number used
as if it were measured. Two inputs settle it, and both are cheap:

  config.json      the architecture, so parameters can be counted rather than assumed
  model_config.py  the math-fidelity settings, so peak is taken at the rate the
                   matmuls actually execute at, not at the fp8 rate by default

Peak was taken as 1548 TFLOPS for two cards at fp8. If the matmuls run bf16 the
peak halves and the gap halves with it, which changes how much is on the table.

Prints every config key it did not use, because an unaccounted-for parameter
block is the difference between "matmuls are inefficient" and "the FLOP count was
wrong", and silently ignoring keys would hide that.
"""

import argparse
import io
import json


def mlp_params(hidden, intermediate, gated=True):
    """gate + up + down for a gated MLP, or up + down for a plain one."""
    return (3 if gated else 2) * hidden * intermediate


def attention_params(hidden, heads, kv_heads, head_dim):
    """q, k, v, o projections."""
    q = hidden * heads * head_dim
    k = hidden * kv_heads * head_dim
    v = hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    return q + k + v + o


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='the model config.json')
    parser.add_argument('--tokens', type=int, default=2568 + 10248 + 20488,
                        help='tokens in the profiled prefill, run 35422536834')
    parser.add_argument('--device-ms', type=float, default=3696.8,
                        help='chip-0 device time for that prefill')
    parser.add_argument('--arithmetic-ms-low', type=float, default=1459.2)
    parser.add_argument('--arithmetic-ms-high', type=float, default=1589.5)
    parser.add_argument('--json')
    options = parser.parse_args()

    config = json.load(io.open(options.config, encoding='utf-8'))
    # Qwen3.5 ships a multimodal wrapper; the language model lives under
    # text_config and the top level carries only vision and token ids.
    config = config.get('text_config', config)
    used = set()

    def get(*names, **kw):
        for name in names:
            if name in config:
                used.add(name)
                return config[name]
        return kw.get('default')

    hidden = get('hidden_size')
    layers = get('num_hidden_layers')
    intermediate = get('intermediate_size')
    heads = get('num_attention_heads')
    kv_heads = get('num_key_value_heads', default=heads)
    head_dim = get('head_dim', default=(hidden // heads if hidden and heads else None))
    vocab = get('vocab_size')
    interval = get('full_attention_interval', default=4)
    gated_attn = get('attn_output_gate', default=False)
    tied = get('tie_word_embeddings', default=False)
    # Gated delta-net mixer dimensions.
    lin_k_heads = get('linear_num_key_heads')
    lin_k_dim = get('linear_key_head_dim')
    lin_v_heads = get('linear_num_value_heads')
    lin_v_dim = get('linear_value_head_dim')
    conv_kernel = get('linear_conv_kernel_dim', default=0)

    report = {'hidden': hidden, 'layers': layers, 'intermediate': intermediate,
              'heads': heads, 'kv_heads': kv_heads, 'head_dim': head_dim,
              'vocab': vocab, 'tokens': options.tokens,
              'full_attention_interval': interval, 'attn_output_gate': gated_attn}

    if not all((hidden, layers, intermediate, heads, head_dim, vocab)):
        report['error'] = 'config lacks the fields needed to count parameters'
        report['config_keys'] = sorted(config.keys())
        print(json.dumps(report, indent=2, default=str))
        return 1

    full_layers = layers // interval if interval else layers
    gdn_layers = layers - full_layers
    report['full_attention_layers'] = full_layers
    report['gdn_layers'] = gdn_layers

    per_mlp = mlp_params(hidden, intermediate)
    per_attn = attention_params(hidden, heads, kv_heads, head_dim)
    if gated_attn:
        # an output gate the width of q
        per_attn += hidden * heads * head_dim

    if all((lin_k_heads, lin_k_dim, lin_v_heads, lin_v_dim)):
        k_width = lin_k_heads * lin_k_dim
        v_width = lin_v_heads * lin_v_dim
        per_gdn = (hidden * k_width * 2      # q and k
                   + hidden * v_width        # v
                   + v_width * hidden)       # output
        per_gdn += conv_kernel * (2 * k_width + v_width)
        report['gdn_sized_from'] = 'its own config fields'
    else:
        per_gdn = per_attn
        report['gdn_sized_from'] = 'ATTENTION AS A STAND-IN - approximate'

    report['per_layer_mlp_m'] = round(per_mlp / 1e6, 1)
    report['per_layer_attn_m'] = round(per_attn / 1e6, 1)
    report['per_layer_gdn_m'] = round(per_gdn / 1e6, 1)

    body = layers * per_mlp + full_layers * per_attn + gdn_layers * per_gdn
    embed = vocab * hidden
    total = body + embed * (1 if tied else 2)
    report['params_body_b'] = round(body / 1e9, 2)
    report['params_total_b'] = round(total / 1e9, 2)

    # Prefill FLOPs: 2 per multiply-accumulate over the body. Embeddings are a
    # lookup and the LM head runs per request, not per prefill token.
    flops = 2.0 * body * options.tokens
    report['prefill_pflop'] = round(flops / 1e15, 3)

    # TP2 splits the work; the profile is one chip.
    per_card = flops / 2.0
    report['weights'] = 'bfloat8_b'
    report['activations'] = 'bfloat16'
    for label, tflops in (('lofi_774_per_card', 774e12), ('hifi2_387_per_card', 387e12)):
        ideal_ms = 1e3 * per_card / tflops
        entry = {'ideal_ms': round(ideal_ms, 1),
                 'device_time_multiple': round(options.device_ms / ideal_ms, 2),
                 'mfu_in_arithmetic_window_pct': [
                     round(100 * ideal_ms / options.arithmetic_ms_high, 1),
                     round(100 * ideal_ms / options.arithmetic_ms_low, 1)]}
        # A rate is impossible if its ideal exceeds the time the arithmetic ops
        # actually occupied: the work cannot take less time than its own floor.
        entry['possible'] = ideal_ms <= options.arithmetic_ms_high
        report[label] = entry

    report['unused_config_keys'] = sorted(set(config.keys()) - used)
    print(json.dumps(report, indent=2, default=str))
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
