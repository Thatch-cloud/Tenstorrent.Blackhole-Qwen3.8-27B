# Long-context drafter study (tp4/drafter-long)

Status: SIMULATED. Everything below is computed on a GPU host from teacher-forced features of Swift-1.5 on its own greedy text; nothing here ran on a card. Numbers marked "card" are the aggregate paired
card result of the v1 drafter that this study set out to explain; they are quoted, not reproduced. No weights, features or prompts are in this repository.

## Summary

* Calibration (gate): PARTIAL. On 66 held-out sequences from 16k to 123k tokens the simulator reproduces the card's pooled v1 gain (x1.026 simulated against x1.017 on cards) and the agent-turn tau level (control 5.42
  against 5.2-5.6 at 16k-64k on cards), but NOT the card's length pattern: the simulated control tau is flat with prompt length (4.5 to 4.9) and v1's ratio is x1.02-1.03 in every bucket including 128k, where the cards saw x0.938. Rotary positions, feature
  noise up to 40% and the window size do not produce the card pattern. No simulated statement here predicts the card's 80k+ result.
* Levers, simulated tau ratio on the control drafter (95% interval): (c) commits across the 256-row edge x1.011 [1.010, 1.013]; (d) prompt lookup as built x1.014 [0.999, 1.041], but over the whole prompt (d2) x1.046 [1.012, 1.089]
  (the 32,768-token index limit removes the gain beyond 32k); (b) two-candidate verify x1.050 [1.045, 1.055] with a second drafter chain and x1.100 [1.063, 1.149] with a lookup branch, for twice the verify rows;
  (a) a full-context window collapses without training (x0.543 [0.506, 0.583]: the drafter's attention dilutes over rows it never saw) and, with a drafter trained on contexts of up to 34k tokens, is x1.061 [1.040, 1.090] at 16k, x1.029 [0.999, 1.064] at 32k, x0.989 [0.965, 1.025] at 64k and x0.926 [0.909, 0.948] at 128k against the same weights with the 2,048-row window. Stacked with (c) and (d2) the second drafter chain gives x1.124 [1.087, 1.169].
* v3 (v1 recipe plus 78 (+10 for validation) long sequences): x1.057 [1.049, 1.067] against the served drafter on the long set and x1.031 [1.023, 1.040] against v1, equal to v1 on the short set (x1.051 against x1.053 at 16 rows).
  The gain over v1 does not grow with length, so it is not evidence about long contexts as such. With (c), (d2) and the second chain the simulated ladder reaches x1.181 [1.141, 1.232].

## Why

The v1 fine-tune (docs/tp4-swift-drafter.md) simulated +27% tau on agent turns and measured x1.017 on cards (182 paired turns with identical text, 95% interval 0.996-1.046), and x0.938 in the 80k+ prompt bucket. The
simulator's held-out set was short (61 sequences, prompts up to 2,064 tokens, inside the drafter's 2,048-row window), so it could not see anything a long context does. Card control tau by prompt bucket (T16) was 5.61 at 16k,
5.46 at 32k, 5.16 at 64k, 4.15 at 80k and 4.66 above; production prompts average about 59k tokens. This study (1) builds a held-out set at 16k to 128k from public data, (2) checks whether the simulator
now reproduces the card result, (3) simulates four levers on it and (4) trains a long-context fine-tune (v3).

## The long held-out set (public data only)

* Prompts: 66 agent-like coding prompts from 12 public open-source repositories (Python, TypeScript, Go, Rust, C and C++, Java) that are disjoint from every repository in the v1 training prompts (the group is the
  repository). A prompt is Swift's chat template with thinking on: a coding-agent system prompt and tool schemas (two styles: dedicated read/search/list/git tools, or a single shell tool), a task taken from a real
  recent commit subject of the repository, and a history of tool calls whose results are real repository content: numbered file reads (up to 600 lines), code searches, directory listings, `git log` and `git show` output,
  until the rendered prompt reaches the target length. The last turn is the agent continuing after a tool result (30), a request for the exact edits (19) or a request for a summary of what was found (17).
* Lengths: 20 prompts of 15.8k-16.3k tokens, 20 of 27.1k-32.7k, 16 of 63.7k-65.4k and 10 of 120.8k-123.8k ("16k", "32k", "64k", "128k" below).
* Generation: Swift-1.5 (revision b4c84d42903a), online fp8 weights, bf16 KV cache, bf16 GDN state, greedy, thinking on, at most 2,048 new tokens (the v1 study's generation setup): 54,370 generated tokens, mean 823 per
  prompt (12 stopped by the limit). Agent turns average 499 tokens, edit and summary answers 1,094.
* Features: a teacher-forced bf16 pass of Swift (HF transformers, 8,192-row chunks with the GDN and convolution state carried over; the outputs of layers 5, 19, 33, 47 and 61 at every position of the whole sequence,
  not just the last 2,064 prompt rows), through the served drafter's `fc` with its weights in the cards' bfloat8_b format, and `hidden_norm`: bf16 [rows, 5120]. 98.6% of the response tokens equal Swift-bf16's own argmax
  (the rest are the fp8 generator's near-ties); every number is computed against both the generated tokens and the bf16 argmax, they differ in the third digit and the generated-token numbers are shown.
* Chunked and single-pass extraction differ by bf16 noise, not by a state error: inside the first 8,192 rows the two are identical, and past a chunk boundary the relative difference is 0.001 / 0.004 / 0.017 / 0.07 / 0.13
  at taps 5 / 19 / 33 / 47 / 61, the same size as changing the chunk size from 8,192 to 4,096 on rows that sit in the first chunk of both.

## The simulator

The v1 simulator (docs/tp4-swift-drafter.md) with these changes; the drafter is run at 16 rows (T16, the card control), bidirectional inside the block, window 2,048, bfloat8_b emulation on every projection:

* Keys are sliced to each chunk of anchors' own window, so a 123k-token sequence costs seconds. Checked against the unsliced reference: identical accepted lengths on all 1,503 anchors of three short held-out sequences;
  97.9% identical (2 of 93 anchors, means 3.011 against 3.032) on a 32k-token sequence, where the two differ only by bf16 reduction order.
* Rotary positions are the ABSOLUTE positions of the sequence (up to 125k); the v1 store had cropped them.
* The round walk applies the packed block's commit cap: a round that starts at position p may commit at most the rows left in the 256-key extent of p (`accept_limit` = min(rows, E - p), E the next multiple of 256 above p),
  which is what the packed verify block of this branch does (its `accept_limit`). Lever (c) removes it.
* Exact walk, tau = committed tokens per round; the ratio of two arms is computed on the same sequences, with a paired bootstrap over sequences (4,000 resamples). Sequences from the same repository are not
  independent (12 repositories); resampling repositories instead gives practically the same intervals for the paired ratios, because the repository effect cancels in the pairing:

| ratio | by sequences (66) | by repositories (12) |
| --- | --- | --- |
| v1 / served drafter | x1.026 [1.019, 1.033] | x1.026 [1.020, 1.033] |
| v3 / served drafter | x1.057 [1.049, 1.067] | x1.057 [1.047, 1.070] |
| v3 / v1 | x1.031 [1.023, 1.040] | x1.031 [1.022, 1.042] |
| (c) edge cap lifted (control) | x1.011 [1.010, 1.013] | x1.011 [1.010, 1.012] |
| (b1) second chain (control) | x1.050 [1.045, 1.055] | x1.050 [1.044, 1.056] |
| (d) lookup as built (control) | x1.014 [0.999, 1.041] | x1.014 [1.002, 1.031] |
| (d2) lookup whole prompt (control) | x1.046 [1.012, 1.089] | x1.046 [1.008, 1.095] |
| (b3) lookup branch (control) | x1.100 [1.063, 1.149] | x1.100 [1.057, 1.160] |

## Calibration against the cards

| drafter | all (n=66) | 16k (n=20) | 32k (n=20) | 64k (n=16) | 128k (n=10) | agent turns (n=30) | long answers (n=36) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| served drafter (control) | 4.68 [4.40, 5.04] | 4.54 [4.25, 4.86] | 4.68 [4.18, 5.44] | 4.92 [4.17, 5.96] | 4.48 [4.17, 4.84] | 5.42 [4.42, 6.49] | 4.44 [4.27, 4.64] |
| v1 | 4.80 [4.50, 5.19] | 4.62 [4.29, 5.00] | 4.81 [4.30, 5.62] | 5.07 [4.26, 6.25] | 4.60 [4.26, 4.96] | 5.65 [4.55, 6.84] | 4.54 [4.35, 4.73] |

Card, for comparison: control tau 5.61 / 5.46 / 5.16 / 4.15 / 4.66 at 16k / 32k / 64k / 80k / above; v1 pooled x1.017 (0.996-1.046), x0.938 at 80k+.

| candidate / control | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 / control | x1.026 [1.019, 1.033] | x1.018 [1.003, 1.035] | x1.026 [1.015, 1.039] | x1.031 [1.017, 1.049] | x1.025 [1.012, 1.039] | x1.043 [1.028, 1.057] | x1.021 [1.014, 1.028] |

What reproduces: the pooled ratio (x1.026 [1.019, 1.033] simulated against x1.017 on cards, inside the card interval), a gain that is larger on agent turns than on long answers (as the v1 study found), and a control tau of
5.42 on agent turns, close to the card level (the long answers, which think at length, are lower). What does NOT reproduce: the card's fall of control tau with length (the simulated control is flat,
4.5 to 4.9 from 16k to 128k; a per-sequence fit has a slope of +0.02 tau per doubling of the prompt) and the card's v1 loss in the longest bucket (simulated x1.025 [1.012, 1.039] at 128k; the per-sequence slope of the ratio is +0.003
per doubling). So whatever produces the card's long-bucket pattern is not something this simulator models.

Diagnostics that tried to produce the card's pattern inside the simulator (same sequences, control and v1 each time):

| arm | control tau, all | v1 tau, all | v1 / control, all | v1 / control, 128k |
| --- | --- | --- | --- | --- |
| baseline (absolute rotary positions, clean features) | 4.675 | 4.796 | x1.026 [1.019, 1.033] | x1.025 [1.012, 1.039] |
| rotary positions relative to the window start | 4.684 | 4.800 | x1.025 [1.017, 1.033] | x1.021 [1.005, 1.034] |
| features + 15% relative noise | 4.673 | 4.782 | x1.023 [1.017, 1.031] | x1.022 [1.009, 1.037] |
| features + 40% relative noise | 4.602 | 4.715 | x1.025 [1.018, 1.031] | x1.018 [0.998, 1.034] |

None of them moves the ratio: absolute rotary positions do not matter, features survive 40% relative noise (control loses 1.6%, v1 keeps its gain), and the drafter's tau does not depend on how much prompt sits behind the
window. How much the window itself matters (control drafter, window in rows, no retraining):

| drafter window (rows) | control tau, all | 16k | 32k | 64k | 128k |
| --- | --- | --- | --- | --- | --- |
| 256 | 4.11 | 4.01 | 4.10 | 4.44 | 3.81 |
| 512 | 4.32 | 4.22 | 4.34 | 4.66 | 3.98 |
| 1024 | 4.54 | 4.44 | 4.52 | 4.79 | 4.36 |
| 2048 | 4.68 | 4.54 | 4.68 | 4.92 | 4.48 |
| 4096 | 4.75 | 4.62 | 4.71 | 5.00 | 4.59 |
| 8192 | 3.57 | 3.32 | 3.58 | 3.84 | 3.48 |
| full | 2.54 | 3.03 | 2.85 | 2.44 | 2.02 |

Reading: tau grows with the window up to 4,096 rows and collapses beyond it, because softmax over keys the drafter never saw at that distance dilutes its attention; the drafter was trained at 2,048.

What the calibration therefore supports, and what it does not. It supports the simulated size of the v1 gain on text of this kind at every length up to 123k (about x1.02-1.03, agent turns x1.04). It does not explain the
card's long-bucket loss, so the simulator cannot be used to promise that a long-context fine-tune removes it. Candidates outside the simulator: the 80k+ card bucket is a subset of the 182 paired turns, so its interval is wide; real 80k+ sessions
are not synthetic transcripts of repository files (content, repetition, tool-output shapes) and reach 262k, past the 123k tested here; the cards run the target with their own numerics, whose long-context features this
simulator replaces with bf16 ones. The paired card measurement of any candidate on 80k+ prompts stays the judge.

## Levers (control drafter, 66 sequences, T16; SIMULATED ratio of tau to the baseline, paired bootstrap)

Definitions: (a) the drafter reads the whole context instead of the 2,048-row window (here with the unchanged weights: a drafter trained for it is below); (b1) two-candidate verify with a second drafter chain (the
selector's second-best choice at the position of its smallest winning margin, then greedy), the longer accepted prefix commits, 32 verify rows instead of 16; (b2/b3) two-candidate verify of the drafter's block and a
prompt-lookup branch (key of 3 tokens, any match, completed with the drafter's own rows); (c) commits across the 256-row extent edge; (d) prompt lookup as in docs/tp4-lookup.md (n3m12: the last 3 tokens are looked up, the match is extended
backwards, at 12 or more the lookup's rows replace the drafter's) with the index limited to the last 32,768 prompt tokens as built (d), or over the whole prompt (d2); (d3) the same with the longest earlier match. Lookup
proposals depend only on the true history, so the walk is exact. b1 and b2 double the verifier's rows, and their gain is tau only: the round time they add is not modelled.

| policy | all (n=66) | 16k (n=20) | 32k (n=20) | 64k (n=16) | 128k (n=10) | agent turns (n=30) | long answers (n=36) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: sliding window 2,048, commit capped at the 256-row edge: tau | 4.675 | 4.539 | 4.684 | 4.918 | 4.484 | 5.424 | 4.445 |
| (c) commits across the 256-row edge | x1.011 [1.010, 1.013] | x1.010 [1.007, 1.013] | x1.013 [1.010, 1.016] | x1.010 [1.007, 1.012] | x1.012 [1.008, 1.015] | x1.012 [1.008, 1.015] | x1.011 [1.009, 1.012] |
| (a) full-context drafter window | x0.543 [0.506, 0.583] | x0.668 [0.653, 0.683] | x0.609 [0.557, 0.649] | x0.496 [0.420, 0.558] | x0.451 [0.438, 0.467] | x0.510 [0.426, 0.592] | x0.554 [0.516, 0.597] |
| (b1) two-candidate verify: drafter chain + its second chain | x1.050 [1.045, 1.055] | x1.057 [1.049, 1.067] | x1.050 [1.042, 1.059] | x1.046 [1.035, 1.058] | x1.049 [1.040, 1.057] | x1.046 [1.038, 1.057] | x1.051 [1.045, 1.057] |
| (b2) two-candidate verify: drafter chain + prompt-lookup branch | x1.047 [1.028, 1.074] | x1.046 [1.035, 1.062] | x1.095 [1.044, 1.186] | x1.028 [1.008, 1.059] | x1.012 [1.003, 1.027] | x1.060 [1.035, 1.087] | x1.043 [1.022, 1.078] |
| (b3) two-candidate verify: drafter chain + lookup branch, whole prompt indexed | x1.100 [1.063, 1.149] | x1.046 [1.035, 1.062] | x1.095 [1.044, 1.186] | x1.131 [1.046, 1.271] | x1.125 [1.051, 1.226] | x1.098 [1.068, 1.135] | x1.101 [1.053, 1.165] |
| (d) prompt lookup n3m12 as built (last 32,768 prompt tokens indexed) | x1.014 [0.999, 1.041] | x0.998 [0.986, 1.007] | x1.051 [1.002, 1.139] | x1.003 [0.990, 1.024] | x0.999 [0.995, 1.006] | x1.014 [0.996, 1.035] | x1.014 [0.996, 1.048] |
| (d2) prompt lookup n3m12, whole prompt indexed | x1.046 [1.012, 1.089] | x0.998 [0.986, 1.007] | x1.051 [1.002, 1.139] | x1.065 [1.001, 1.184] | x1.066 [1.000, 1.149] | x1.017 [0.999, 1.036] | x1.056 [1.012, 1.110] |
| (d3) prompt lookup n3 gate 12, longest match, whole prompt indexed | x1.057 [1.013, 1.111] | x1.000 [0.988, 1.012] | x1.060 [1.003, 1.169] | x1.077 [0.997, 1.233] | x1.090 [1.001, 1.210] | x1.019 [1.001, 1.037] | x1.070 [1.014, 1.141] |
| (c)+(d2) | x1.057 [1.022, 1.100] | x1.007 [0.995, 1.017] | x1.062 [1.012, 1.153] | x1.075 [1.012, 1.193] | x1.079 [1.014, 1.163] | x1.027 [1.009, 1.044] | x1.067 [1.023, 1.123] |
| (c)+(b1) | x1.061 [1.056, 1.066] | x1.066 [1.057, 1.075] | x1.063 [1.055, 1.071] | x1.056 [1.044, 1.068] | x1.062 [1.049, 1.073] | x1.057 [1.049, 1.066] | x1.063 [1.056, 1.069] |
| (c)+(b3) | x1.113 [1.074, 1.162] | x1.056 [1.044, 1.073] | x1.109 [1.056, 1.205] | x1.141 [1.056, 1.280] | x1.142 [1.066, 1.245] | x1.108 [1.077, 1.148] | x1.114 [1.066, 1.179] |
| (c)+(d2)+(b1) | x1.124 [1.087, 1.169] | x1.076 [1.066, 1.085] | x1.129 [1.074, 1.233] | x1.139 [1.075, 1.263] | x1.146 [1.084, 1.235] | x1.088 [1.074, 1.105] | x1.135 [1.088, 1.195] |
| (c)+(d3)+(b1) | x1.140 [1.093, 1.198] | x1.083 [1.072, 1.093] | x1.143 [1.076, 1.270] | x1.158 [1.078, 1.321] | x1.175 [1.088, 1.303] | x1.092 [1.079, 1.108] | x1.156 [1.095, 1.233] |

Prompt-lookup sensitivity (ratio to the baseline; key n, gate m):

| lookup policy (key n, gate m, matcher, index) | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n3m12, most recent, last 32,768 prompt tokens | x1.014 [0.999, 1.041] | x0.998 [0.986, 1.007] | x1.051 [1.002, 1.139] | x1.003 [0.990, 1.024] | x0.999 [0.995, 1.006] | x1.014 [0.996, 1.035] | x1.014 [0.996, 1.048] |
| n3m12, most recent, whole prompt | x1.046 [1.012, 1.089] | x0.998 [0.986, 1.007] | x1.051 [1.002, 1.139] | x1.065 [1.001, 1.184] | x1.066 [1.000, 1.149] | x1.017 [0.999, 1.036] | x1.056 [1.012, 1.110] |
| n3m12, longest, whole prompt | x1.057 [1.013, 1.111] | x1.000 [0.988, 1.012] | x1.060 [1.003, 1.169] | x1.077 [0.997, 1.233] | x1.090 [1.001, 1.210] | x1.019 [1.001, 1.037] | x1.070 [1.014, 1.141] |
| n2m12, most recent, whole prompt | x1.039 [1.009, 1.073] | x0.999 [0.989, 1.007] | x1.044 [1.003, 1.118] | x1.052 [0.999, 1.145] | x1.055 [1.000, 1.131] | x1.017 [1.000, 1.036] | x1.045 [1.008, 1.091] |
| n4m12, most recent, whole prompt | x1.050 [1.012, 1.097] | x0.997 [0.986, 1.006] | x1.054 [1.002, 1.148] | x1.070 [1.001, 1.201] | x1.077 [1.000, 1.180] | x1.018 [1.000, 1.037] | x1.061 [1.012, 1.122] |
| n6m12, most recent, whole prompt | x1.052 [1.012, 1.101] | x0.998 [0.988, 1.007] | x1.053 [1.000, 1.150] | x1.072 [0.999, 1.211] | x1.081 [1.000, 1.188] | x1.018 [1.000, 1.037] | x1.063 [1.013, 1.127] |
| n3m8, most recent, whole prompt | x1.045 [1.012, 1.086] | x0.998 [0.987, 1.008] | x1.042 [0.993, 1.128] | x1.067 [1.002, 1.181] | x1.072 [1.010, 1.151] | x1.024 [1.004, 1.042] | x1.052 [1.010, 1.106] |
| n3m24, most recent, whole prompt | x1.045 [1.011, 1.085] | x1.002 [0.995, 1.008] | x1.050 [1.004, 1.137] | x1.056 [0.997, 1.169] | x1.068 [1.004, 1.158] | x1.012 [0.998, 1.030] | x1.056 [1.012, 1.110] |
| n3m4, most recent, whole prompt | x1.024 [0.990, 1.066] | x0.977 [0.967, 0.989] | x1.015 [0.972, 1.089] | x1.049 [0.973, 1.166] | x1.054 [0.981, 1.142] | x1.025 [0.997, 1.055] | x1.024 [0.982, 1.077] |
| n3m4, most recent, last 32,768 prompt tokens | x0.984 [0.967, 1.007] | x0.977 [0.967, 0.989] | x1.015 [0.972, 1.089] | x0.970 [0.945, 1.006] | x0.968 [0.948, 0.991] | x1.004 [0.980, 1.032] | x0.978 [0.958, 1.007] |

Reading: (c) is worth a steady +1.1% at no verify cost. (b1) is worth +5% for twice the verify rows; with the lookup branch (b3) +10%, +13% at 64k and 128k. The lookup index limit of 32,768 prompt tokens removes the whole lookup gain beyond 32k
(x1.003 at 64k and x0.999 at 128k with the limit, x1.065 and x1.066 over the whole prompt), because the useful matches are far back, beyond the drafter's window: indexing the whole prompt costs about 0.27 s per 123k tokens at
admission (the figure measured in docs/tp4-lookup.md). The full-window arm with unchanged weights collapses (see the window table): any window above about 4,096 needs training, which the next section does. The levers stack roughly additively because they fix different misses. The same table for
the v1 weights is within a point of these (v1: (c) x1.010, (b1) x1.044, (d2) x1.043).

## Lever (a), trained: a full-context window

With the unchanged weights a window above about 4,096 rows collapses (table above: whole context x0.54), so the lever needs a drafter trained for it. v3fw is v3's recipe with the sliding window removed: the drafter attends to the whole context
(every row before the anchor) in training and in the simulation. Training needs each sequence's whole context, which the long pool stores only for prompts up to 34k tokens, so v3fw trains on the 703 v1-corpus sequences that keep their
whole context and the 65 long sequences of at most 34k tokens (prompts of 8k to 34k). The 64k and 128k buckets are therefore EXTRAPOLATION for it: it never saw a context longer than 34k rows.

Validation during training (mean accepted rows at 16 rows):

| step | full window: short slice | full window: long validation slice | sliding window (v3): short slice | sliding window (v3): long validation slice |
| --- | --- | --- | --- | --- |
| 0 | 4.154 | 2.732 | 4.154 | 4.582 |
| 50 | 4.662 | 4.544 | 4.819 | 4.843 |
| 100 | 5.173 | 5.137 | 5.283 | 4.920 |
| 150 | 5.492 | 5.251 | 5.550 | 5.016 |
| 200 | 5.527 | 5.389 | 5.667 | 5.086 |
| 250 | 5.729 | 5.442 | 5.727 | 5.125 |
| 300 | 5.793 | 5.427 | 5.781 | 5.156 |

Held-out long set, tau at 16 rows and ratios (SIMULATED):

| drafter | all (n=66) | 16k (n=20) | 32k (n=20) | 64k (n=16) | 128k (n=10) | agent turns (n=30) | long answers (n=36) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| served drafter (control) | 4.68 [4.40, 5.04] | 4.54 [4.25, 4.86] | 4.68 [4.18, 5.44] | 4.92 [4.17, 5.96] | 4.48 [4.17, 4.84] | 5.42 [4.42, 6.49] | 4.44 [4.27, 4.64] |
| v3, sliding window 2,048 | 4.94 [4.64, 5.34] | 4.79 [4.42, 5.23] | 4.96 [4.43, 5.78] | 5.24 [4.47, 6.39] | 4.70 [4.33, 5.10] | 5.79 [4.72, 6.97] | 4.69 [4.49, 4.89] |
| v3fw weights, whole context | 4.87 [4.57, 5.27] | 5.04 [4.67, 5.50] | 5.01 [4.47, 5.82] | 5.07 [4.30, 6.25] | 4.32 [4.01, 4.64] | 5.84 [4.75, 6.94] | 4.59 [4.38, 4.82] |
| v3fw weights, window 2,048 | 4.87 [4.58, 5.25] | 4.75 [4.40, 5.16] | 4.87 [4.35, 5.68] | 5.13 [4.35, 6.27] | 4.66 [4.33, 5.05] | 5.68 [4.64, 6.81] | 4.63 [4.43, 4.83] |

| candidate / control | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3, sliding window 2,048 / served drafter | x1.057 [1.049, 1.067] | x1.054 [1.037, 1.080] | x1.058 [1.047, 1.076] | x1.066 [1.050, 1.089] | x1.048 [1.028, 1.070] | x1.067 [1.052, 1.091] | x1.054 [1.044, 1.065] |
| v3fw weights, whole context / served drafter | x1.042 [1.022, 1.066] | x1.110 [1.093, 1.141] | x1.071 [1.043, 1.108] | x1.031 [1.009, 1.069] | x0.963 [0.947, 0.987] | x1.076 [1.051, 1.119] | x1.032 [1.007, 1.060] |
| v3fw weights, window 2,048 / served drafter | x1.042 [1.035, 1.051] | x1.047 [1.032, 1.065] | x1.040 [1.033, 1.052] | x1.043 [1.026, 1.066] | x1.040 [1.025, 1.063] | x1.047 [1.032, 1.067] | x1.041 [1.033, 1.050] |

| candidate / v3 | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3fw weights, whole context / v3 | x0.986 [0.968, 1.007] | x1.053 [1.033, 1.077] | x1.011 [0.983, 1.043] | x0.967 [0.948, 1.000] | x0.919 [0.905, 0.931] | x1.009 [0.982, 1.043] | x0.978 [0.957, 1.003] |
| v3fw weights, window 2,048 / v3 | x0.986 [0.979, 0.993] | x0.993 [0.978, 1.004] | x0.983 [0.973, 0.991] | x0.978 [0.961, 0.993] | x0.992 [0.975, 1.009] | x0.982 [0.965, 0.995] | x0.987 [0.979, 0.995] |

The window effect alone, with the weights held fixed:

| candidate / the same weights with window 2,048 | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3fw weights: whole context / window 2,048 | x1.000 [0.980, 1.023] | x1.061 [1.040, 1.090] | x1.029 [0.999, 1.064] | x0.989 [0.965, 1.025] | x0.926 [0.909, 0.948] | x1.027 [0.997, 1.069] | x0.991 [0.969, 1.017] |

For reference, untrained wider windows on the served drafter:

| candidate / control | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| served drafter, whole context, untrained | x0.543 [0.506, 0.583] | x0.668 [0.653, 0.683] | x0.609 [0.557, 0.649] | x0.496 [0.420, 0.558] | x0.451 [0.438, 0.467] | x0.510 [0.426, 0.592] | x0.554 [0.516, 0.597] |
| served drafter, window 4,096, untrained | x1.016 [1.005, 1.028] | x1.018 [0.997, 1.042] | x1.006 [0.987, 1.029] | x1.017 [0.998, 1.048] | x1.024 [1.004, 1.047] | x1.014 [0.993, 1.048] | x1.016 [1.004, 1.029] |
| served drafter, window 8,192, untrained | x0.764 [0.747, 0.781] | x0.732 [0.713, 0.746] | x0.764 [0.726, 0.796] | x0.781 [0.738, 0.829] | x0.776 [0.760, 0.794] | x0.763 [0.715, 0.815] | x0.764 [0.749, 0.781] |

Reading. A drafter trained to read the whole context gains where it was trained: with the same weights, reading the whole context instead of 2,048 rows is worth x1.061 [1.040, 1.090] at 16k and x1.029 [0.999, 1.064] at 32k,
breaks even at 64k (x0.989 [0.965, 1.025]) and loses 7% at 128k (x0.926 [0.909, 0.948]), where its attention has to cover three to four times more rows than any context it saw in training (at most 34k). Against v3 (sliding window)
the full-window drafter is x1.053 [1.033, 1.077] at 16k and x0.919 [0.905, 0.931] at 128k, x0.986 [0.968, 1.007] overall: its own sliding-window reading (x1.042 against the served drafter) is also weaker than v3's (x1.057), plausibly because it
trained on fewer long sequences (65 against 78) and none above 34k rows. Whether a drafter trained on 64k to 128k contexts keeps the 16k gain at those lengths is NOT answered here: that needs whole-context features of
longer sequences than the long pool stored and a training pass that attends to up to 125k keys. The gain is tau only: the draft KV of a whole-context window grows with the prompt (the cards' drafter keeps 2,048 rows) and the draft
attention cost with it, neither of which is modelled.

## v3: the v1 recipe with long-context sequences

Recipe. Initialised from the served drafter, the v1 settings throughout (frozen `fc`, `hidden_norm`, embedding and head; AdamW lr 2e-5 with 30 warm-up steps and a cosine to 10%, 300 steps of 4 sequences, generated-token
labels, bidirectional in-block rule, blocks of 4, 8 and 16 rows with probability .25 / .25 / .5, up to 8,192 query rows per sequence). What changes: 30% of the sequences of a step come from a second pool of 78 (+10 for validation) long agent-like
sequences (same public-repository construction as the held-out set, other repositories: 28 repositories, of which 4 are held out for validation; prompts of 8k to 123k tokens, generated by Swift-1.5 the same way,
answers of 954 tokens on average, edits, tests and summaries weighted up), the rest from the v1 corpus (815 sequences); the rotary positions are absolute (the long sequences start at their true position);
the context features come from the long pass (bfloat8_b `fc`). The sliding window stays 2,048 rows, so a long prompt matters through what its target features look like and what is asked of the drafter, not through reach.

Validation during training (fixed slices, mean accepted rows at 16 rows; the served drafter scores 4.154 and 4.582):

| step | short held-out slice (12 sequences), mean accepted rows | long validation slice (10 sequences of held-out repositories), mean accepted rows |
| --- | --- | --- |
| 0 | 4.154 | 4.582 |
| 50 | 4.819 | 4.843 |
| 100 | 5.283 | 4.920 |
| 150 | 5.550 | 5.016 |
| 200 | 5.667 | 5.086 |
| 250 | 5.727 | 5.125 |
| 300 | 5.781 | 5.156 |

Held-out long set (66 sequences, tau at 16 rows, ratio to the served drafter, SIMULATED):

| drafter | all (n=66) | 16k (n=20) | 32k (n=20) | 64k (n=16) | 128k (n=10) | agent turns (n=30) | long answers (n=36) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| served drafter (control) | 4.68 [4.40, 5.04] | 4.54 [4.25, 4.86] | 4.68 [4.18, 5.44] | 4.92 [4.17, 5.96] | 4.48 [4.17, 4.84] | 5.42 [4.42, 6.49] | 4.44 [4.27, 4.64] |
| v1 | 4.80 [4.50, 5.19] | 4.62 [4.29, 5.00] | 4.81 [4.30, 5.62] | 5.07 [4.26, 6.25] | 4.60 [4.26, 4.96] | 5.65 [4.55, 6.84] | 4.54 [4.35, 4.73] |
| v3 | 4.94 [4.64, 5.34] | 4.79 [4.42, 5.23] | 4.96 [4.43, 5.78] | 5.24 [4.47, 6.39] | 4.70 [4.33, 5.10] | 5.79 [4.72, 6.97] | 4.69 [4.49, 4.89] |

| candidate / control | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 / served drafter | x1.026 [1.019, 1.033] | x1.018 [1.003, 1.035] | x1.026 [1.015, 1.039] | x1.031 [1.017, 1.049] | x1.025 [1.012, 1.039] | x1.043 [1.028, 1.057] | x1.021 [1.014, 1.028] |
| v3 / served drafter | x1.057 [1.049, 1.067] | x1.054 [1.037, 1.080] | x1.058 [1.047, 1.076] | x1.066 [1.050, 1.089] | x1.048 [1.028, 1.070] | x1.067 [1.052, 1.091] | x1.054 [1.044, 1.065] |

Paired against v1:

| candidate / v1 | all | 16k | 32k | 64k | 128k | agent turns | long answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 / v1 | x1.031 [1.023, 1.040] | x1.035 [1.020, 1.057] | x1.031 [1.017, 1.052] | x1.034 [1.015, 1.056] | x1.022 [1.012, 1.038] | x1.023 [1.007, 1.049] | x1.033 [1.025, 1.043] |

The v1 study's short held-out set (61 sequences, bfloat8_b emulation, generated-token truth), ratio to the served drafter:

| drafter | 4 rows | 8 rows | 16 rows |
| --- | --- | --- | --- |
| v1 | x1.014 [1.011, 1.018] | x1.023 [1.016, 1.032] | x1.053 [1.043, 1.067] |
| v3 | x1.012 [1.009, 1.015] | x1.018 [1.012, 1.026] | x1.051 [1.042, 1.064] |

Reading. v3 beats v1 on the long set by x1.031 [1.023, 1.040] and does not lose on the short set (x1.051 against x1.053 at 16 rows, inside the intervals). The gain over v1 is the same at every length (x1.035, x1.031, x1.034, x1.022 from 16k
to 128k), larger on edits and summaries (x1.033) than on agent turns (x1.023). It is therefore NOT evidence that long contexts as such were the problem: the long pool differs from the v1 corpus in what it contains (tool-call
transcripts of real repository files, long code answers) as well as in how much prompt sits in front, and this design cannot separate the two (that needs the same sequences with their prompts shortened). Because the
simulator does not reproduce the card's loss of v1 in the longest bucket, a simulated v3 gain also says nothing about whether v3 removes it; the paired card measurement on 80k+ prompts is the test.

Everything stacked, against the served drafter as built (control weights, commit cap, no lookup), SIMULATED, tau only:

| arm | tau, all | x vs served: all | x vs served: 16k | x vs served: 32k | x vs served: 64k | x vs served: 128k |
| --- | --- | --- | --- | --- | --- | --- |
| served drafter, as built | 4.675 | 1 | 1 | 1 | 1 | 1 |
| v1 weights | 4.796 | x1.026 [1.019, 1.033] | x1.018 [1.003, 1.035] | x1.026 [1.015, 1.039] | x1.031 [1.017, 1.049] | x1.025 [1.012, 1.039] |
| v3 weights (long-context fine-tune) | 4.943 | x1.057 [1.049, 1.067] | x1.054 [1.037, 1.080] | x1.058 [1.047, 1.076] | x1.066 [1.050, 1.089] | x1.048 [1.028, 1.070] |
| v3 + (c) commits across the 256-row edge | 4.996 | x1.069 [1.060, 1.079] | x1.063 [1.046, 1.088] | x1.071 [1.058, 1.091] | x1.077 [1.061, 1.100] | x1.059 [1.039, 1.082] |
| v3 + (c) + (d2) prompt lookup over the whole prompt | 5.211 | x1.115 [1.077, 1.162] | x1.062 [1.044, 1.083] | x1.127 [1.075, 1.221] | x1.136 [1.062, 1.271] | x1.127 [1.049, 1.230] |
| v3 + (c) + (d2) + (b1) second drafter chain (2x verify rows) | 5.523 | x1.181 [1.141, 1.232] | x1.128 [1.108, 1.154] | x1.188 [1.133, 1.292] | x1.209 [1.132, 1.352] | x1.194 [1.111, 1.305] |
| v3 + (c) + (b3) lookup branch as second candidate (2x verify rows) | 5.475 | x1.171 [1.127, 1.227] | x1.110 [1.082, 1.149] | x1.173 [1.119, 1.271] | x1.208 [1.109, 1.374] | x1.188 [1.088, 1.314] |

## Limits

* One generator (a script that assembles agent-style transcripts from public repositories) and 12 repositories; the response mix (30 agent turns averaging 499 tokens, 36 long answers averaging 1,094) is not the card traffic's.
* The card long-bucket loss is not reproduced, so no statement here is a prediction for it.
* Teacher-forced bf16 features stand in for the card target's features; the generator is fp8 (about 1.4% of tokens differ from the bf16 argmax).
* Lever gains are tau only; two-candidate verify and the lookup index have costs that only a card measures.
