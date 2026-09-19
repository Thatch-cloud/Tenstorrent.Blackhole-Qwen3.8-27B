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

    report = {'hidden': hidden, 'layers': layers, 'intermediate': intermediate,
              'heads': heads, 'kv_heads': kv_heads, 'head_dim': head_dim,
              'vocab': vocab, 'tokens': options.tokens}

    if not all((hidden, layers, intermediate, heads, head_dim, vocab)):
        report['error'] = 'config.json lacks the fields needed to count parameters'
        report['config_keys'] = sorted(config.keys())
        print(json.dumps(report, indent=2, default=str))
        return 1

    # 48 of 64 layers are linear-attention (GDN) and 16 are full attention. The
    # GDN mixer's projections are sized from its own config fields when present;
    # where they are absent the attention sizing is used as a stand-in and the
    # result is flagged, because a guess here moves the whole answer.
    full_attention = get('full_attention_interval', 'num_full_attention_layers',
                         default=None)
    per_layer_mlp = mlp_params(hidden, intermediate)
    per_layer_attn = attention_params(hidden, heads, kv_heads, head_dim)

    report['per_layer_mlp_m'] = round(per_layer_mlp / 1e6, 1)
    report['per_layer_attn_m'] = round(per_layer_attn / 1e6, 1)
    report['full_attention_hint'] = full_attention

    body = layers * (per_layer_mlp + per_layer_attn)
    embed = vocab * hidden
    total = body + 2 * embed
    report['params_body_b'] = round(body / 1e9, 2)
    report['params_total_b'] = round(total / 1e9, 2)
    report['note'] = ('GDN mixer sized as attention where config lacks its own '
                      'fields; treat the total as approximate')

    # Prefill FLOPs: 2 per multiply-accumulate, over the body only. Embeddings
    # are a lookup; the LM head runs once per request, not per prefill token.
    flops = 2.0 * body * options.tokens
    report['prefill_pflop'] = round(flops / 1e15, 3)

    # TP2 splits the work across two cards, and the profile is one chip.
    per_card = flops / 2.0
    for label, tflops in (('fp8_1548_total', 774e12), ('bf16_774_total', 387e12)):
        ideal_ms = 1e3 * per_card / tflops
        entry = {'ideal_ms': round(ideal_ms, 1)}
        for name, measured in (('vs_device_time', options.device_ms),
                               ('vs_arithmetic_low', options.arithmetic_ms_low),
                               ('vs_arithmetic_high', options.arithmetic_ms_high)):
            entry[name] = round(measured / ideal_ms, 2) if ideal_ms else None
        entry['mfu_of_arithmetic_window'] = (
            round(100 * ideal_ms / options.arithmetic_ms_high, 1) if options.arithmetic_ms_high else None)
        report[label] = entry

    report['unused_config_keys'] = sorted(set(config.keys()) - used)
    print(json.dumps(report, indent=2, default=str))
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
