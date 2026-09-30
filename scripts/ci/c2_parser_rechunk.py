"""R21 fix M: re-chunk multi-token deltas at the parsers' marker tokens (C2 serving, API server).

C2 commits 3-16 tokens per engine step (DFlash, 15 speculative tokens), so vLLM's streaming chat
path hands DelegatingParser.parse_delta multi-token deltas (serving.py:596). The qwen3 reasoning
and qwen3_xml tool parsers (one ParserEngine grammar, qwen3.py:79-189) then diverge from what
they produce at one token per step (the general profile, and every stock vLLM stream):
- K1 (semantic): a delta that holds a real tag id (</think>, <tool_call>) and a later lookalike of
  the same tag spelled as BPE pieces binds the tag to the lookalike, because
  TokenIDScanner._rebuild_from_anchors resolves anchors right-to-left with rfind
  (token_id_scanner.py:256-268); the real tag's text is then content (reasoning keeps
  "</think>", or "<tool_call>" leaks into content). No delta-size threshold: 5 of 40 random
  1-16 schedules hit it in the R21 corpus.
- K2 (whitespace): whitespace-only content after a tool call is dropped per chunk
  (parser_engine.py:741-761), so a delta that carries it together with text keeps it.

M feeds the parser what it sees at one token per step exactly where that matters, and the whole
delta everywhere else:
- a delta that carries a marker token id - the ids the parsers' TokenIDScanner matches, read
  from the parser's own engines (StreamingParserEngine._resolved_token_ids: <think> 248068,
  </think> 248069, <tool_call> 248058, </tool_call> 248059, plus every special id, e.g. EOS
  248046, as a drop terminal) - is fed one token per sub-delta;
- so is every delta in the whitespace window after a </tool_call> and before the first content
  that is not whitespace (K2's two residual schedules in the R21 prototype);
- any other delta, in particular the long runs inside tool arguments whose re-conversion is
  quadratic per call (parser_engine.py:930-936), stays whole.
The sub-results merge into ONE DeltaMessage per engine step. Sub-delta text is each token's
text as the engine's FastIncrementalDetokenizer renders it (detokenizer.py:167-246), from a
private tokenizers DecodeStream kept in lockstep with the engine's; every delta's pieces must
join to exactly the engine's delta_text, or M switches off for that request.

Hardening (verified-r21-parsers.md, Corrections 4):
- the private DecodeStream's step is guarded as the engine guards its own
  (detokenizer.py:223-246): OverflowError/TypeError give no text, tokenizers' "Invalid prefix
  encountered" resets the stream (unprimed, as the engine does) and steps again;
- anything else M cannot handle - another decode error, pieces that do not join to the engine's
  delta_text (a stop string held back, a request the engine detokenizes differently) - switches
  M off for THAT request, which continues on the stock path from that delta on; the switch-off is
  logged once per request (lengths only, never text). Nothing M adds raises out of parse_delta;
  the parser's own exceptions propagate as they would unpatched;
- the merge sets only fields that are not None, as ParserEngine._events_to_delta builds its
  messages (parser_engine.py:765-773), so model_dump_json(exclude_unset=True) (serving.py:743)
  never carries "role":null or "content":null;
- the private stream is primed with the prompt's last PRIME_TOKENS ids, not all of them: the
  first step of a stream primed with a 131,072-token prompt decodes the whole prompt twice
  (about 125 ms on the dev PC, on the API server's event loop); from the first text it emits,
  a tail-primed stream holds the same state as a fully primed one.

NOT included: the I1 fix (a tool call cut off by max_tokens closed into valid JSON). Alone it
hides the truncation while finish_reason stays "tool_calls" (serving.py:694-695), so an agent
would execute a write of a truncated file; unpatched, the stream carries invalid JSON a client
can detect. It ships only together with finish_reason "length", which this module does not do.

Installed only in the vLLM API server (the parsers run there, not in the EngineCore), by
serving_c2_contract.boot for a profile with "parser_rechunk": true (c2 and c2-gate), through a
PostImportHook on vllm.parser.abstract_parser: DelegatingParser.parse_delta (vLLM 0.25.1
abstract_parser.py:793) is wrapped. ParserManager.get_parser returns a subclass of it that does
not override parse_delta (parser_manager.py:128-137), and one wrapper covers chat
(serving.py:596) and Responses (responses/serving.py:1368). Markers: one line at install, one
the first time a delta is actually re-chunked (graft mounted is not graft executed), one per
switch-off. Python 3.10 (the image), stdlib only; tokenizers is imported lazily, at the first
delta of a request.
"""

import inspect
import sys

MODULE = 'vllm.parser.abstract_parser'
CLASS = 'DelegatingParser'
# vllm/v1/engine/detokenizer.py:27 (tokenizers mod.rs DecodeStreamError::InvalidPrefix).
INVALID_PREFIX = 'Invalid prefix encountered'
# detokenizer.py:22-24, 60-65: only tokenizers >= 0.22 detokenizes through a primed DecodeStream.
FAST_DETOKENIZER = (0, 22)
PRIME_TOKENS = 64
TOOL_START, TOOL_END = 'TOOL_START', 'TOOL_END'
MAX_SWITCH_OFF_LINES = 50
PARAMETERS = ('self', 'delta_text', 'delta_token_ids', 'request', 'prompt_token_ids', 'finished')
INSTALLED = '_qwen_c2_parser_rechunk'
STATE = '_qwen_c2_parser_rechunk_state'

STATS = dict(requests=0, rechunked=0, windowed=0, switched_off=0, inactive=0)
_LOGGED = set()


def log(message, *values):
    try:
        sys.stderr.write('[QWEN-C2] parser M: ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def _log_once(key, message, *values):
    if key not in _LOGGED:
        _LOGGED.add(key)
        log(message, *values)


def install(module, emit=None):
    """Wrap module.DelegatingParser.parse_delta once. Refuses (raises) a module whose class or
    signature is not vLLM 0.25.1's: a profile that asks for M must not serve without it."""
    emit = log if emit is None else emit
    parser_cls = getattr(module, CLASS, None)
    if not isinstance(parser_cls, type):
        raise RuntimeError('%s has no %s class: parser M cannot be installed' % (module.__name__, CLASS))
    if parser_cls.__dict__.get(INSTALLED):
        return parser_cls
    original = parser_cls.__dict__.get('parse_delta')
    if original is None:
        raise RuntimeError('%s.%s defines no parse_delta: parser M cannot be installed' % (module.__name__, CLASS))
    signature = inspect.signature(original)
    names = tuple(signature.parameters)
    if names != PARAMETERS or signature.parameters['finished'].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError('%s.%s.parse_delta%s is not the vLLM 0.25.1 signature: parser M cannot be installed'
                           % (module.__name__, CLASS, signature))
    parser_cls.parse_delta = rechunked(original)
    setattr(parser_cls, INSTALLED, True)
    emit('installed on %s.%s.parse_delta: a multi-token delta is fed one token per sub-delta when it carries a '
         'marker token or falls in the whitespace window after a tool call, and the sub-results merge into one '
         'DeltaMessage per engine step' % (module.__name__, CLASS))
    return parser_cls


def rechunked(original):
    """A parse_delta that re-chunks, around the stock one (original)."""

    def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished):
        attributes = getattr(self, '__dict__', None)
        stream = attributes.get(STATE) if attributes is not None else INACTIVE
        if stream is None:
            stream = open_stream(self, request, prompt_token_ids)
            attributes[STATE] = stream
        plan = stream.plan(delta_text, delta_token_ids, finished) if stream.on else None
        if plan is None:
            result = original(self, delta_text, delta_token_ids, request, prompt_token_ids, finished=finished)
        else:
            last = len(plan) - 1
            result = stream.merge([original(self, text, ids, request, prompt_token_ids, finished=finished and i == last)
                                   for i, (text, ids) in enumerate(plan)], type(self).__name__, len(plan))
        if stream.on:
            stream.observe(result)
        return result

    parse_delta.__wrapped__ = original
    parse_delta.__doc__ = original.__doc__
    return parse_delta


class Stream(object):
    """One request's M state: the private decode stream and the whitespace window."""

    on = True

    def __init__(self, decode_stream, raw, markers, prompt_token_ids, skip_special_tokens,
                 spaces_between_special_tokens):
        self.decode_stream, self.raw, self.markers = decode_stream, raw, markers
        self.skip_special_tokens = skip_special_tokens
        self.spaces = skip_special_tokens or spaces_between_special_tokens
        self.added = None if self.spaces else added_token_contents(raw)
        self.last_special = False
        tail = list(prompt_token_ids or ())[-PRIME_TOKENS:]
        self.decoder = decode_stream(ids=tail, skip_special_tokens=skip_special_tokens)
        self.tool_seen = self.in_tool = self.nonws_content = False

    def step(self, token_id):
        """The engine's FastIncrementalDetokenizer.decode_next for one token."""
        try:
            piece = self.decoder.step(self.raw, token_id)
        except (OverflowError, TypeError):
            piece = None
        except Exception as error:
            if not str(error).startswith(INVALID_PREFIX):
                raise
            # detokenizer.py:236-246: a fresh, unprimed stream, stepped again.
            self.decoder = self.decode_stream(skip_special_tokens=self.skip_special_tokens)
            piece = self.decoder.step(self.raw, token_id)
        if not self.spaces:
            special = self.added.get(token_id)
            if special is not None and self.last_special:
                piece = special
            self.last_special = special is not None
        return piece or ''

    def plan(self, delta_text, delta_token_ids, finished):
        """[(text, [token id])] to feed one token at a time, or None for the whole delta."""
        try:
            ids = list(delta_token_ids)
            if not ids:
                return None
            pieces = [self.step(token_id) for token_id in ids]
            if ''.join(pieces) != delta_text:
                if finished and ''.join(pieces[:-1]) == delta_text:
                    pieces[-1] = ''  # a stop token: the engine leaves its text out (detokenizer.py:107-113)
                else:
                    return self.switch_off('its %d-token delta is %d characters, the private decode stream '
                                           'rendered %d' % (len(ids), len(delta_text), len(''.join(pieces))))
            window = self.tool_seen and not self.in_tool and not self.nonws_content
            marked = False
            for token_id in ids:
                terminal = self.markers.get(token_id)
                if terminal is None:
                    continue
                marked = True
                if terminal == TOOL_START:
                    self.tool_seen = self.in_tool = True
                elif terminal == TOOL_END:
                    self.in_tool = False
            if len(ids) == 1 or not (marked or window):
                return None
            STATS['rechunked'] += 1
            if not marked:
                STATS['windowed'] += 1
            return [(piece, [token_id]) for piece, token_id in zip(pieces, ids)]
        except Exception as error:
            return self.switch_off('the private decode stream failed: %s: %s' % (type(error).__name__, error))

    def switch_off(self, reason):
        self.on = False
        self.decoder = None
        return switched_off(reason)

    def observe(self, message):
        if not self.nonws_content and message is not None:
            content = getattr(message, 'content', None)
            if isinstance(content, str) and content.strip():
                self.nonws_content = True

    def merge(self, messages, class_name, tokens):
        _log_once(('live', class_name), 'live: first re-chunked delta in %s (%d tokens)', class_name, tokens)
        try:
            return merge(messages)
        except Exception as error:
            # The sub-deltas are consumed: the stock path cannot take this delta again. Keep what
            # the parser produced, uncoalesced if need be (a chunk may list an index twice, as
            # DelegatingParser._flush_engine_parsers's own concatenation can).
            self.switch_off('merging sub-results failed: %s: %s' % (type(error).__name__, error))
            try:
                return merge(messages, coalesce_calls=False)
            except Exception:
                return next((message for message in reversed(messages) if message is not None), None)


class Inactive(object):
    """M does not apply to this parser (not an engine parser, no markers, no fast tokenizer)."""

    on = False


INACTIVE = Inactive()


def open_stream(parser, request, prompt_token_ids):
    """A Stream for this request, or INACTIVE (logged once per reason) when M cannot apply."""
    STATS['requests'] += 1
    try:
        if not getattr(parser, '_engine_based', False):
            return inactive(parser, 'not an engine-based parser')
        markers = marker_terminals(parser)
        if not markers:
            return inactive(parser, 'its parser engines match no token ids')
        raw = getattr(getattr(parser, 'model_tokenizer', None), '_tokenizer', None)
        if raw is None or not hasattr(raw, 'decode'):
            return inactive(parser, 'its tokenizer has no fast (tokenizers) backend')
        decode_stream = decode_stream_class()
        if decode_stream is None:
            return inactive(parser, 'tokenizers < %d.%d: the engine detokenizes without a DecodeStream'
                            % FAST_DETOKENIZER)
        return Stream(decode_stream, raw, markers, prompt_token_ids,
                      bool(getattr(request, 'skip_special_tokens', True)),
                      bool(getattr(request, 'spaces_between_special_tokens', True)))
    except Exception as error:
        switched_off('opening the private decode stream failed: %s: %s' % (type(error).__name__, error))
        return INACTIVE


def switched_off(reason):
    """Count a request's switch-off and log it, the first MAX_SWITCH_OFF_LINES of them."""
    STATS['switched_off'] += 1
    count = STATS['switched_off']
    if count <= MAX_SWITCH_OFF_LINES:
        log('switched off for a request, which continues on the stock path: %s (%d so far)%s', reason, count,
            '; further switch-offs are counted, not logged' if count == MAX_SWITCH_OFF_LINES else '')
    return None


def inactive(parser, reason):
    STATS['inactive'] += 1
    _log_once((type(parser).__name__, reason), 'inactive for %s: %s', type(parser).__name__, reason)
    return INACTIVE


def marker_terminals(parser):
    """{token id: terminal} the parser's own engines match by id: TokenIDScanner's map
    (streaming_parser_engine.py:118-141), from the reasoning and the tool adapter's engines."""
    found = {}
    for adapter in (getattr(parser, '_reasoning_parser', None), getattr(parser, '_tool_parser', None)):
        engine = getattr(getattr(adapter, '_parser_engine', None), '_engine', None)
        resolved = getattr(engine, '_resolved_token_ids', None)
        if isinstance(resolved, dict):
            found.update(resolved)
    return found


def decode_stream_class():
    """tokenizers.decoders.DecodeStream, looked up now as the engine does (so a backend shim that
    rebinds it is honoured), or None where the engine uses the slow detokenizer."""
    try:
        import tokenizers
        import tokenizers.decoders
    except ImportError:
        return None
    if version_tuple(getattr(tokenizers, '__version__', '0')) < FAST_DETOKENIZER:
        return None
    return getattr(tokenizers.decoders, 'DecodeStream', None)


def version_tuple(text):
    parts = []
    for part in str(text).split('.')[:2]:
        digits = ''
        for character in part:
            if not character.isdigit():
                break
            digits += character
        parts.append(int(digits or 0))
    return tuple(parts + [0] * (2 - len(parts)))


def added_token_contents(raw):
    """{id: content} of the added tokens, as FastIncrementalDetokenizer builds it
    (detokenizer.py:193-205), for spaces_between_special_tokens=False."""
    return {token_id: token.content for token_id, token in raw.get_added_tokens_decoder().items()}


def merge(messages, coalesce_calls=True):
    """One DeltaMessage from the sub-deltas' results: content and reasoning concatenated, tool-call
    deltas coalesced per index as ParserEngine._coalesce_tool_call_deltas does, and only fields
    that are not None set."""
    messages = [message for message in messages if message is not None]
    if not messages:
        return None
    if len(messages) == 1:
        return messages[0]
    fields = {}
    role = next((message.role for message in messages if getattr(message, 'role', None) is not None), None)
    if role is not None:
        fields['role'] = role
    for name in ('content', 'reasoning'):
        parts = [getattr(message, name) for message in messages if getattr(message, name, None) is not None]
        if parts:
            fields[name] = ''.join(parts)
    calls = [call for message in messages for call in (getattr(message, 'tool_calls', None) or ())]
    if calls:
        fields['tool_calls'] = coalesce(calls) if coalesce_calls else calls
    return type(messages[0])(**fields)


def coalesce(calls):
    """ParserEngine._coalesce_tool_call_deltas (parser_engine.py:891-918): one entry per index, in
    first-seen order; later entries fill id, type and name if unset and append arguments. Like it,
    this assigns only values that are not None, so a field that was unset stays unset."""
    merged = {}
    for call in calls:
        existing = merged.get(call.index)
        if existing is None:
            merged[call.index] = call
            continue
        if call.id is not None and existing.id is None:
            existing.id = call.id
        if call.type is not None and existing.type is None:
            existing.type = call.type
        if call.function is not None:
            if existing.function is None:
                existing.function = call.function
            else:
                if call.function.name is not None and existing.function.name is None:
                    existing.function.name = call.function.name
                if call.function.arguments is not None:
                    if existing.function.arguments is None:
                        existing.function.arguments = call.function.arguments
                    else:
                        existing.function.arguments += call.function.arguments
    if len(merged) == len(calls):
        return list(calls)
    return list(merged.values())
