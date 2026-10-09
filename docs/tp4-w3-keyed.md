# tp4/w3 lever 1b: the keyed staging diff (QWEN_FAST_TP4_PRESTAGE_KEYED)

Status: **built, CPU-tested, not run on a card.** Default off, byte-identical off, gate-only. Expected saving **-1.1 to -1.5 ms a round (ESTIMATED**, eight-seat host-gap design 1b: the verify-time diff's `packed_values` rebuild of about 150 values, 1.38 ms p50 at four seats, is skipped on about 75% of four-live blocks).

## What it changes

The verify pre-stage (`QWEN_FAST_PRESTAGE`, `verify_prestage.py`) writes every verify input but the tokens inside the drafts' fence window and keeps a snapshot of what it wrote. At verify time the **value diff** still rebuilds all ~150 staging values on the host (`packed_verifier.packed_values`: rotary tables, per-row page tables, the readers' words and tables, each reader's checks) only to find that everything but the tokens equals the snapshot.

Every one of those values is a pure function of each segment's `(start, page table)` and of the block's fixed structure (rotary constants, the destinations, the readers). So the snapshot now also keeps a **key**:

- per segment, the start and a **clone** of the `(1, page_width)` host table (the engine's own table is refreshed in place at execute time, after the window);
- the identity of every destination holder (the fixture's token, position, rotary and page buffers, its per-row buffers, its cache tiles, the replay reader and each segment reader with its position, table and current-position buffers) and the numbers the values are sliced by (each reader's capacity, page width and bundle count).

A verify whose key equals the snapshot's builds only the tokens and writes the tokens buffer alone. A key that differs in anything (a start, one table entry, the live set, the padded idle fill, a destination, a reader's geometry) takes today's value diff, unchanged. An epoch bump still sends the verify to the full stage first (the keyed path is only reached with a usable snapshot).

## What the keyed path still checks (host only, before any copy)

All the checks the full build makes, in the same order and with the same helpers:

- the fixture checks (`packed_context`, factored out of `packed_values`);
- the tokens and the page tables per user (`packed_host_tokens`, factored out of `packed_host_inputs`: **the one place the tokens are built**, called by the full stage, the pre-stage, the value diff and the keyed path);
- the T2 K/V guard (`kv_guard`, factored out of `packed_values`) on the snapshot's own positions and tables, which the key makes the verify-time ones, including its audit line;
- every reader's `validate(start)`;
- the tokens buffer's shape, dtype and layout against the fixture's.

The write is `write_packed` with the tokens alone, `fence=False`, the same poison/bump semantics as the diff (`bump('verify-failed')` on a failed copy, `bump_fixture('verify')` after, each reader's `start` set). No fence is added or removed.

## Exactness

Equal inputs to a deterministic function give equal outputs: the non-token values the device holds from the pre-stage are the ones the verify-time build would produce. Anything not in the key is a block constant fixed at attach (the rotary base and dimension, the shape, the dtypes and layouts); a destination or reader replaced after attach changes the structure part of the key and falls back to the value diff (then, as today, to the full stage on a destination mismatch). The output tokens therefore cannot change; the audits below are how a wrong construction would be caught on a card.

## Audits

- `QWEN_FAST_PRESTAGE_AUDIT=1` **or** the full audit (`QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1`): on a keyed round the full `packed_values` is built on the host and **every non-token value** is held against the snapshot's, destinations included. One line per audited keyed round, `[PACKED-PRESTAGE-KEYED-AUDIT] round=N checked=K keyed_mismatches=M`; a difference restages the round in full before the trace, so the round stays exact, and fails the arm in the smoke.
- The full audit also reads every destination back from every chip after the keyed write (its line ends ` keyed=1`) and then stages the round in full anyway, as for any diff round: the audited twin proves the keyed write left the device holding the full stage, and its trace sees the control's inputs.
- The existing rotating device audit (`QWEN_FAST_PRESTAGE_AUDIT`) still runs beside it.

## Markers (new text only with the flag on)

| Line | When |
|---|---|
| `[PINDIAG] verify prestage keyed engaged users=U audit=A` | once per block at attach |
| `[PACKED-PRESTAGE] round=N path=diff buffers=1 reason=- live=L keyed=1` | on a keyed verify (the line is today's otherwise) |
| `[PACKED-HOSTGAP-VERIFY] ... readback_cpu_ms=X keyed=0\|1` | with `QWEN_FAST_TP4_HOSTGAP_LOG=1` |
| `[PACKED-PRESTAGE-KEYED-AUDIT] round=N checked=K keyed_mismatches=M` | audited keyed rounds |
| `[PACKED-PRESTAGE-FULLAUDIT] ... keyed=1` | full audit of a keyed round |

The existing parsers (`c2_smoke_check.HOSTGAP_VERIFY_LINE`, `HOSTGAP_PRESTAGE_LINE`, `lever_n_m3native_gate.PRESTAGE_LINE`, `HOSTGAP_FULL_AUDIT`) are unanchored at the line end and still match; a test holds this.

## Smoke rule

`scripts/ci/w3_keyed_smoke.py` (`problems(env, container_text)`, stdlib only, host side): a profile without the flag logs none of the lever's text; with it the engaged line must exist, at least 30% of the `path=diff` verifies must be keyed once 20 diff rounds are logged (the design expects about 75% of four-live blocks), every keyed audit line must read `keyed_mismatches=0`, and an audited arm must log at least one keyed audit line.

## Flag grammar

`QWEN_FAST_TP4_PRESTAGE_KEYED` is strict `0|1`; any other value raises when the block is built and again at the attach of several blocks (`engage_two_block`). It needs `QWEN_FAST_PRESTAGE=1` (the image sets it); alone it keys nothing. It is deliberately not in `HOSTGAP_FLAGS` (the host-gap family the older suites enumerate). It composes with the two-block pre-stage and the per-block epochs (the key is per fixture) but does not require them.

## What only a card can show

- The saving itself: `diff_ms` of the keyed rounds against the value diff's (`[PACKED-PRESTAGE-WINDOW]` and `[PACKED-HOSTGAP-VERIFY]` `input_ms`), paired per round in the ABAB; the keyed share of the diff rounds on real traffic (the 75% is a model: a user crosses a page about every 14.7 rounds at tau 4.35, and any user crossing makes the block unkeyed).
- Zero `keyed_mismatches` and zero full-audit mismatches on the audited twin over the real shapes (equal, skewed and 262k contexts, padded rounds, Lever N prefill chunks between rounds).
- That the readers' real classes (the extent readers and the multi-user SDPA twin) expose the attributes the structure part of the key reads; an attribute a class lacks is read as absent at both ends, so the key is weaker for it, never wrong, and the audit covers the values.

## Tests

`scripts/ci/test_tp4_w3_keyed.py` (flag grammar, flag-off identity, keyed rounds over random schedules against the value diff and the full stage, every key breaker, the checks that still run, the audits including a rotary base the key does not see, the extent readers' block, markers and existing parsers, the smoke rule, hygiene).
