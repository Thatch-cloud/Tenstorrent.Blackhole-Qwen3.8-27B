# tp4-levern-floor-jobs: the decode-gap floor test (one short card window, 2026-10-09)

Lever N's alternation failed the hang shapes twice on 2026-10-08 (HL-LN v584: a 199 s stretch without a decode round; v620: 241 back-to-back prefill step pairs). Branch `tp4/levern-alternation-fix` adds the decode-gap floor
(`QWEN_FAST_LEVERN_MAX_DECODE_GAP_S=8`, baked into the Lever N traffic profile) and a smoke rule that measures the whole decode-less stretch. This pack is the card test of that fix and nothing else: no digest, no audit, no timed block.

| Job | Class | What |
|---|---|---|
| `A0X0` | stop | the agent stops, the serving container goes, a rescan, the reset of all four cards |
| `HLF` | soft | one fresh boot of `tp4-serve-12b` on the traffic profile, the nine hang-shape tests; a failure is data and never strands the hand-back |
| `ZR`, `LM`, `TICK`, `Z` | hand / drv | the hand-back as the session 0808b pack ends: reset, link re-measure, topology wait, the agent started; the operator then runs the release step |

What `HLF` must show (READ): the `lever N governor: ttft=180 gap_floor=8` line in the container log; `levern_alternation_problems` clean (no decode-less stretch above about 16 s); `concurrent8_skew` slowest first token under 238 s with the shorts
keeping their round; every user at budget, no refusal, no hang. A floor-off comparison on production bytes (an HL on `tp4-serve-10`) is NOT in this pack: it does not fit a 70 minute window.

Clock: READY target minute 70, soft cap 100, hard cap 120 from the driver's start (`A0X0` = minute 0). Tags `v597`, `v598`, `v599`, `v601`, reserve `v602`, `v603`, `v606`. Never run beside the session packs of 2026-10-08.
