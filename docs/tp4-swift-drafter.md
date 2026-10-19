# A drafter fine-tuned on Swift-1.5's own text (tp4/swift-drafter)

Status: trained and scored in simulation on a GPU host; nothing here has touched a card. The weights are NOT in this repository (they are the owner's to release, see
"Open decisions"): only the pins, the loader support, the tau lab arm and the card pack are.

## Why

On the cards Swift-1.5 decodes slower than the base (8 users at 4k: 218 against 239 tokens/s, unpaired). The DFlash2 drafter that proposes for it was fit to the BASE's
text and hidden states, so its acceptance on Swift's text may be lower. The target's embedding and LM head follow the target at run time; the drafter's own five layers,
its feature projection and its candidate selector do not.

## What the served drafter is, and the recipe

The served drafter is the public `incoai/Qwen3.8-27B-DFlash2` (short form `dedf8df6`, Apache-2.0, mirrored by z-lab): five sliding-window (2,048) layers of hidden size 5,120,
fed by the outputs of target layers 5, 19, 33, 47 and 61 through one projection (`fc`, 25,600 to 5,120) and an RMS norm, two-tap dynamic grouped convolutions around the
attention and the MLP of each layer, and a candidate selector (rank 256, top 16, two 248,320 x 256 codebooks). It was trained upstream at block size 8; we run it at T16.
No upstream training code is in this repository. `tp4/tau-next` holds our own CPU reference of the recipe (the SpecForge DFlash2 objective: hard cross-entropy on the target's
tokens with a position decay exp(-(k-1)/7), plus the selector's cross-entropy over the strict top 16 of the unary logits, alpha 1), parity-tested against the upstream model code
and the recipe's own loss on tiny configurations (docs/ft-tau-fine-tune.md). The GPU run reuses those functions unchanged (a device-aware copy of the anchor sampler and the rotary
tables) and adds the chunked, recomputed loss and the packed-anchor forward with activation checkpointing; its tests compare it with the reference on CPU.

Two findings from reading the sources carry over: the recipe's training mask is CAUSAL inside a block for a sliding-window draft, while the serving path (and z-lab's inference model)
reads the block BIDIRECTIONALLY. This run trains and scores both ways (see Results).

## The data (corpus A, v1)

Public prompt sets only; no SWE-rebench (the tau lab's own held-out set), no own-session or customer text. Prompts were rendered with Swift's chat template (thinking on, the template default),
and Swift-1.5 (revision `b4c84d42903a`) answered them greedily:

| set (revision) | licence | use |
| --- | --- | --- |
| SWE-bench/SWE-smith-trajectories (`08e109b4a59e`) | MIT | agent turns: the conversation up to a random assistant turn, Swift writes that turn |
| SWE-Gym/OpenHands-Sampled-Trajectories (`baf3a4e4bff5`) | none stated on the hub | the same, with tool schemas and tool calls through the chat template |
| m-a-p/CodeFeedback-Filtered-Instruction (`a08c213a9748`) | Apache-2.0 | single prompts, and two-turn chains (its answer, then one of eight fixed follow-ups) |
| ise-uiuc/Magicoder-OSS-Instruct-75K (`5f839b1f368a`) | MIT | the same |
| nvidia/OpenCodeInstruct (`8f3ba5bafe4d`) | CC-BY-4.0 | the same |
| bigcode/self-oss-instruct-sc2-exec-filter-50k (`356bb069eee8`) | ODC-By | the same |

Held out BY GROUP (repository for the agent traces, prompt for the rest; 6% of groups, 14-25% for the traces): 61 sequences, scored by nothing that trains. Corpus A is 1,001 sequences and
1.23 million generated tokens (agent turns 0.10 M, code answers 0.38 M, chained answers 0.75 M; answers stop at 2,048 tokens for agent turns and 3,072 for the rest, 26% of them by that limit).
Caveat: the two agent-trace sets draw on public repositories that may also appear in the tau lab's SWE set; the lab's held-out list was not available to this work.

Generation ran Swift with online fp8 weights, bf16 KV and a bf16 GDN state (the speed the host allows); the FEATURES are a teacher-forced pass of Swift in bf16 (fp32 GDN state). The two agree on
97.8% of the 1.16 million response tokens (top-1), and the first token bf16 would not have emitted comes at a median of 36 response tokens (chained), 53 (code) and 12 (agent turns): about one token in 45
of the corpus is a near-tie that bf16 greedy would have taken the other way. The stored per-position bf16 argmax is therefore an alternative LABEL (`labels=argmax`), the proxy closest to what the cards emit.

## Training (a 121 GB GPU host, one model resident at a time)

Initialised from the served drafter. Frozen: `fc` and `hidden_norm` (their output is the stored feature, as in the recipe), the target's embedding and LM head. Trained: the five layers, the final norm
and the candidate selector. Features: the outputs of target layers 5, 19, 33, 47 and 61 at every position, through the served `fc` and `hidden_norm`, stored only for the 2,064 rows before the first answer token
(the window the drafter can ever read). Per sequence up to 8,192 query rows: random anchors in the answer, blocks of 4, 8 or 16 rows (probability .25 / .25 / .5) so that every block the cards run
(1 live user: 4 rows, 2-5: 8, 16 for the long block) is trained; the bidirectional in-block rule (the TT read); loss = the recipe's (CE with exp(-(k-1)/7) decay + selector CE, alpha 1). AdamW,
fp32 master weights with bf16 autocast, peak lr 2e-5 (30 warm-up steps, cosine to 10%), weight decay 0, clip 1.0, 4 sequences per step, 300 steps (about 1.5 passes over the 815 usable sequences),
activation checkpointing per layer and per loss chunk. About 4.5 s per sequence on one GPU.

## Results (v1, simulated on the 61 held-out sequences)

Simulator: for every position of every answer, the drafter's block (anchor token + mask tokens, context = the target features strictly before the anchor in the 2,048-row window) goes through the
selector's greedy path; a proposal is accepted while it equals Swift's next token. The ROUND WALK jumps from anchor to anchor by (accepted + 1), so tau = committed tokens per round is exact for each
drafter (the context of an anchor never depends on the walk). Checked against production: the served drafter simulates 3.06 at 4 rows (production 3.46) and 4.17 at 8 rows (production 5.3-6.4 on agent
traffic). Paired bootstrap over sequences (4,000 resamples) for the ratio:


Truth = the generated tokens. tau = committed tokens per round; ratio = candidate / served drafter, paired bootstrap 95% interval.

| drafter | all: 4 rows | 8 rows | 16 rows | agent turns: 4 rows | 8 rows | 16 rows | chained: 16 rows | code: 16 rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| served drafter (control) | 3.064 | 4.165 | 4.495 | 3.298 | 4.637 | 5.085 | 4.185 | 5.052 |
| v1 (final, step 300) | 3.107 (x1.014 [1.010, 1.019]) | 4.261 (x1.023 [1.017, 1.032]) | 4.749 (x1.056 [1.046, 1.069]) | 3.471 (x1.052 [1.030, 1.085]) | 5.435 (x1.172 [1.105, 1.268]) | 6.505 (x1.279 [1.193, 1.408]) | 4.375 (x1.045 [1.033, 1.058]) | 5.285 (x1.046 [1.036, 1.060]) |

Truth = the bf16 argmax tokens. tau = committed tokens per round; ratio = candidate / served drafter, paired bootstrap 95% interval.

| drafter | all: 4 rows | 8 rows | 16 rows | agent turns: 4 rows | 8 rows | 16 rows | chained: 16 rows | code: 16 rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| served drafter (control) | 3.056 | 4.133 | 4.458 | 3.287 | 4.593 | 5.035 | 4.160 | 4.989 |
| v1 (final, step 300) | 3.097 (x1.013 [1.009, 1.018]) | 4.220 (x1.021 [1.015, 1.029]) | 4.690 (x1.052 [1.042, 1.067]) | 3.470 (x1.056 [1.033, 1.087]) | 5.377 (x1.170 [1.106, 1.262]) | 6.452 (x1.281 [1.190, 1.422]) | 4.333 (x1.042 [1.030, 1.054]) | 5.187 (x1.040 [1.030, 1.051]) |

Reading: the gain is largest where the cards' traffic is: agent turns (long copied spans of code and tool output) gain 5% at 4 rows, 17% at 8 rows and 28% at 16 rows; chained and code answers (long thinking text)
gain 4-5% at 16 rows and 1-2% at 4-8 rows. Held-out agent turns are few (32 sequences, 4.5 k answer tokens), so their intervals are wide. The simulator scores bf16 GPU arithmetic: the cards run the projections in
bfloat8_b, which may eat some of a small gain (the tau lab on the cards is the judge). The public block-16 drafter (b16-98759a49) is indistinguishable from the served one on Swift in the same simulator (4.48 against 4.50 at 16 rows).

## Packaging (what this branch adds)

* `drafter-manifests/<name>.json` with `"source": "local"`: a drafter trained here has no hub repository. Its model is `local/<name>`, its revision is the first 40 hex digits of the
  sha256 of the weights file (content-derived), and `base` says what it was trained from. Header, size and the 81 tensor pins are the served layout's, so the same loader checks
  every byte. `drafter_manifest.validate` accepts `source` `hub` (the default) or `local` and refuses a local manifest that is not shaped that way.
* `drafter-configs/<name>.config.json`: the served config with `dflash_config.block_size` 16 (the block size it was trained for; nothing in the TT or vLLM paths reads the key). The image
  build copies it for a local manifest (instead of fetching a hub config) and holds it to the manifest's `config_sha256`.
* `drafter_checkpoints.json`: a pinned candidate entry with the same revision and config hash, the nine fixture manifest hashes and the weights digest of the staged tree, geometry equal to
  the default's. The profile twin that selects it through `QWEN_FAST_DRAFTER_CHECKPOINT` is not written: the card arm uses the manifest route below.
* Tau lab arm `sw1ft-bf8` (c2_serving_job, c2_tau_lab, drafter_pair_report, tau_lab_report). It is judged PAIRED per turn against the `control` arm run on Swift (job keys C2_MODEL_REPO and
  C2_MODEL_REVISION on both), by the pre-registered GO rule of docs/drafter-arms.md.
* `references/fusion-jobs/DR`: the card pack (two builds from one commit, X0, the tau pair T1/T2, the timed decode ABAB C1 L1 C2 L2, the audited attach SA1, Z), `scripts/ci/test_swift_drafter.py`.
* The candidate image is built and smoked, NEVER pushed: the job reader refuses `build push` with a candidate manifest and the push step refuses an image carrying a drafter marker. Shipping the
  fine-tune needs that rule lifted for an approved local manifest, the exactness gates re-run on the candidate image, and the owner's decision.

## What the card window needs

1. The trained checkpoint on the build host (3.85 GB, one wired copy), staged with `python3 scripts/ci/drafter_stage.py stage --checkpoint <file> --manifest <name> --cache <fixture cache>`.
2. Production down (Swift is what the cards serve), the ARC runner sets at zero for the timed jobs, then the pack in `references/fusion-jobs/DR/ORDER.txt` (the tau pair is the decision; the ABAB confirms the round time).

## Open decisions (the owner's)

* Licence: the training text is Swift-1.5's output and the features are its hidden states (Swift Open License; commercial use above its revenue threshold is conditional). Whether a drafter trained
  on them is a derivative work is a reading of that licence, not a technical question.
* The public datasets used as prompts are listed in the training record with their licences (MIT, Apache-2.0, CC-BY-4.0, ODC-By). The two agent-trace sets draw on public repositories that may also
  appear in the tau lab's own SWE set; the lab's held-out set was not available to this work, so the paired GPU numbers are clean (held-out by repository or prompt) and the card tau on A1 may flatter
  the fine-tune by some repository overlap.
