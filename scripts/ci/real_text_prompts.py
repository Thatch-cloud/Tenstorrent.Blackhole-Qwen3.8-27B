"""Real-text chat prompts for the M3native gate: N users, each reading a different part of real code.

Every acceptance and rate figure the programme had was measured on the synthetic prompt
(lever_n_m3native_gate.prompt_for: token ids [base + i % 64]). Its output repeats every 64
tokens, no round ever emitted more than 10 tokens, and draft positions 10-15 never accepted.
This module builds what a real coding request looks like instead, inside the container, with
nothing but the image's own files:

CORPUS. The installed vLLM package source (the directory importlib finds for `vllm`, located
without importing it): every *.py file, sorted by its POSIX path relative to the package's
parent, each prefixed 'File: <relpath>' + newline, joined with blank lines. It is fixed per
image, so a concurrent arm and a sequential arm on the same image build identical prompts. The
runner checkout is NOT a valid corpus: tags sit on different commits. vLLM 0.25.1 is 1967 files,
about 27 M characters; four users at 131k need about 2.25 M.

SPECIAL-TOKEN TEXT. That source contains the chat template's own markers as string literals
('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>', '<|image_pad|>', ...).
apply_chat_template(tokenize=True) would encode each as its special id in the middle of the
prompt. Every added-token text the tokenizer knows (special, or written <...>) is neutralised in
the corpus first by a space after its first character ('<|im_start|>' becomes '< |im_start|>'),
and every built prompt must then carry exactly the template's own multiset of those ids (the
same prompt with an empty excerpt); any other count refuses the build.

WINDOWS. User i reads a disjoint contiguous window of the corpus starting at i * len // N and
ending where user i + 1's starts, so the users read different parts of the codebase. The prompt
is its window's prefix inside make_context_prompt's framing (coding_context_request.py: system
'You are a careful coding assistant.', the <repository_context> block, the chat template with
add_generation_prompt=True and enable_thinking=False). That module is not mounted in the
container (it imports coding_request, which is not either), so the framing is copied here and
test_real_text_prompts checks the two against each other.

TASKS. Each user gets a task that elicits a LONG answer (TASKS[i % 4]), so all users stay active
for the whole measured decode: a user that stops early turns the rounds after it into 3-user
rounds, which leave the packed 64-row step and emit at most four tokens.

LENGTH. Every prompt is EXACTLY `target` tokens, because the image pins it. The serving path
builds each request's session at position len(prompt), and the qualified T16 target attention
refuses any position but the arm's request context (frozen_combined_runtime.
validate_target_option: position == the selected context, 32768 or 131072; run 35442532141 was
refused on exactly that); the packed block also captures one replay family, [T, T + 256). A
shorter prompt is refused in from_prefill. So each prompt is a prefix of its window found by a
secant search on the whole prompt (seeded by a chars-per-token estimate from a sample of the
window, bracketed so it can only converge) that stops at the first prefix encoding to exactly
the target. Byte-level BPE can step over a count (one more character re-merges the last tokens
into two more); then the longest prefix below is used and the few missing tokens are
single-token fillers (FILLER_TEXT, a newline) spliced in at the token level just before the
footer's tokens - the gate sends token ids, so nothing re-tokenises them - at most MAX_PADDING,
recorded per user. The search is deterministic, so both arms build the same prompts. The
tokenizer calls and seconds each user took are recorded (a 131k prompt must not take minutes).

PROVENANCE. Per user: prompt_tokens, the window and excerpt character ranges, the excerpt's
sha256, the prompt's sha256 (sha256 of json.dumps(tokens, separators=(',', ':')), as
coding_context_request records it) and the task index; per corpus: its sha256 before and after
neutralisation, file count and characters. write_prompts puts the whole set, tokens included,
in results/real-text-prompts.json.

A LENGTH PER USER (targets=[...], the C2 serving gate's real-text matrix: once the fast path
takes any prompt length, S1's G4 needs users at {60, 255, 2047, 2048, 2049, 4k, 32k, ~60k,
~120k} in one arm). Opt-in: without `targets` every prompt is `target` long exactly as above,
byte for byte. With it, user i's prompt is exactly targets[i] tokens, from the same disjoint
window of the corpus. A target too short for make_context_prompt's framing (its template is
115-133 tokens on the served tokenizer, plus MIN_EXCERPT_TOKENS of context) takes the COMPACT
framing instead: one user turn, COMPACT_TASKS[i % 4] (stand-alone tasks that still ask for a
long answer), then COMPACT_HEADER and the excerpt, fitted and padded the same way. Each entry
then records its framing and the top level its targets.

The tokenizer (transformers AutoTokenizer on the served snapshot) is imported lazily, so the CPU
tests inject a fake one. Python 3.10 compatible: the container's interpreter is /opt/venv 3.10.
"""

from collections import Counter
from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

SYSTEM = 'You are a careful coding assistant.'
HEADER = ('Repository context excerpt for background only; it may end partway through a file.\n'
          'Do not reproduce or modify this excerpt. Complete only the coding task after it.\n'
          '<repository_context>\n')
FOOTER = '\n</repository_context>\n\nCoding task:\n'

# User i gets TASKS[i % 4]. Each asks for a long answer so the user is still decoding when the
# 256-token budget runs out.
TASKS = (
    'Write pytest unit tests for the last complete function or class in the repository context '
    'above. Cover its normal behaviour, its edge cases and its error handling, with one test '
    'function per behaviour and a one-line docstring on each. Return all of the tests in one '
    'Python code block.',
    'Review the last complete file in the repository context above. For every issue you find '
    '(bugs, unclear names, missing error handling, performance problems), quote the exact code, '
    'explain what is wrong with it, and give a corrected version of that code. Cover every issue '
    'you can find, one after another.',
    'Explain, file by file, what the repository context above does: for each file, its purpose, '
    'its main classes and functions and what each of them does. Then explain how the pieces '
    'interact with each other.',
    'Refactor the last complete function in the repository context above for readability while '
    'keeping its behaviour identical. Give the full refactored function in one Python code block, '
    'then a numbered list of every change you made and why it preserves the behaviour.',
)
TASK_NAMES = ('pytest_tests', 'code_review', 'explain_files', 'refactor_function')

# The compact framing, for a per-user target too short for the repository framing: stand-alone
# tasks (no "repository context above" to point at) that still ask for a long answer, then the
# excerpt. Short, so a 60-token prompt still carries a few tokens of real code.
COMPACT_HEADER = '\n\nRelated code, which may end mid-file:\n'
COMPACT_TASKS = (
    'Write a complete Python LRU cache with per-entry expiry and thread safety, then a full pytest suite.',
    'Implement a JSON parser in pure Python without the json module, then tests for every edge case.',
    'Explain in depth how an asyncio event loop schedules coroutines, with annotated code examples.',
    'Write a token-bucket rate limiter in Python with docstrings, then review its design at length.',
)
COMPACT_TASK_NAMES = ('lru_cache', 'json_parser', 'asyncio_explainer', 'rate_limiter')
MIN_EXCERPT_TOKENS = 64   # of real code below which a per-user target takes the compact framing
FRAMINGS = ('repository', 'compact')

SAMPLE_CHARACTERS = 65536
MAX_CALLS = 40            # per user; a converging search needs a handful
MAX_PADDING = 16          # fillers a prompt may take where BPE steps over the target (0-2 in practice)
FILLER_TEXT = chr(10)     # one token in a byte-level BPE: an extra blank line before the footer
ANGLE = re.compile(r'^<.+>$', re.DOTALL)


def package_root(name='vllm'):
    """The installed package's directory, found without importing the package."""
    spec = importlib.util.find_spec(name)
    locations = list(getattr(spec, 'submodule_search_locations', None) or []) if spec else []
    if not locations:
        raise RuntimeError('%s is not an installed package here' % name)
    return Path(locations[0])


def build_corpus(root):
    """Every *.py under `root`, sorted by POSIX path relative to root's parent, each prefixed
    'File: <relpath>' and a newline, joined with a blank line. Returns (corpus, info)."""
    root = Path(root)
    base = root.parent
    files = sorted((path.relative_to(base).as_posix(), path) for path in root.rglob('*.py') if path.is_file())
    if not files:
        raise ValueError('no *.py files under %s' % root)
    corpus = '\n\n'.join('File: %s\n%s' % (relative, path.read_bytes().decode('utf-8', 'replace'))
                         for relative, path in files)
    return corpus, dict(root=str(root), files=len(files), characters=len(corpus),
                        sha256=sha256_text(corpus), first_file=files[0][0], last_file=files[-1][0])


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prompt_sha256(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def load_tokenizer(model):
    """transformers' AutoTokenizer on the served snapshot (imported here, never at module load),
    loaded local-only and without remote code, like every other in-image client
    (baseline-client.py, dspark-target-hardware.py, full-prefix.py)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)


def filler_token(tokenizer, text=FILLER_TEXT):
    """The one token id `text` encodes to without special tokens; refuses anything else."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    ids = list(ids)
    if len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0:
        raise ValueError('filler %r must encode to exactly one token id, got %r' % (text, ids))
    return ids[0]


def guarded_tokens(tokenizer):
    """{text: id} of every added token the tokenizer matches atomically that a prompt must not
    carry outside the template: the special ones, and any written <...> (Qwen's <think> is an
    added token that is not flagged special)."""
    guarded = {}
    decoder = getattr(tokenizer, 'added_tokens_decoder', None)
    if isinstance(decoder, Mapping) and decoder:
        for token_id, token in decoder.items():
            text = getattr(token, 'content', token)
            if isinstance(text, str) and len(text) > 1 and (getattr(token, 'special', False) or ANGLE.match(text)):
                guarded[text] = int(token_id)
    else:
        getter = getattr(tokenizer, 'get_added_vocab', None)
        vocab = getter() if callable(getter) else {}
        specials = set(getattr(tokenizer, 'all_special_tokens', None) or ())
        for text, token_id in vocab.items():
            if isinstance(text, str) and len(text) > 1 and (text in specials or ANGLE.match(text)):
                guarded[text] = int(token_id)
    return guarded


def neutralise(corpus, texts):
    """The corpus with a space after the first character of every occurrence of each text, and how
    many of each were broken. One regex pass (longest text first, so the longest match wins) per
    round, repeated until nothing matches: a replacement can only split a text, never form one,
    but the loop makes that a checked fact instead of an argument."""
    counts = Counter()
    ordered = sorted({text for text in texts if text}, key=lambda text: (-len(text), text))
    if not ordered:
        return corpus, {}
    pattern = re.compile('|'.join(re.escape(text) for text in ordered))

    def split(match):
        text = match.group(0)
        counts[text] += 1
        return text[0] + ' ' + text[1:]

    for _ in range(8):
        if pattern.search(corpus) is None:
            return corpus, dict(sorted(counts.items()))
        corpus = pattern.sub(split, corpus)
    raise ValueError('special-token text survives neutralisation: %s' % sorted(set(pattern.findall(corpus)))[:5])


def encode_prompt(tokenizer, excerpt, task):
    """make_context_prompt's framing and template call around `excerpt`, as flat token ids."""
    content = HEADER + excerpt + FOOTER + task
    encoded = tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
        return_dict=False, enable_thinking=False)
    return flat_tokens(encoded)


def flat_tokens(encoded):
    tokens = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('Expected nonempty flat token IDs')
    return tokens


def encode_compact(tokenizer, excerpt, task):
    """The compact framing around `excerpt`: one user turn, the task, then the code; the same
    template call as encode_prompt otherwise (generation prompt, thinking off)."""
    encoded = tokenizer.apply_chat_template([
        {'role': 'user', 'content': task + COMPACT_HEADER + excerpt}], tokenize=True, add_generation_prompt=True,
        return_dict=False, enable_thinking=False)
    return flat_tokens(encoded)


def parse_targets(text):
    """A per-user target list from 'L1,L2,...' (or a list of ints): positive integers, at least one."""
    parts = text if isinstance(text, (list, tuple)) else [part.strip() for part in str(text).split(',')]
    targets = []
    for part in parts:
        try:
            value = part if type(part) is int else int(str(part).strip())
        except ValueError:
            raise ValueError('prompt lengths must be comma-separated positive integers, got %r' % (text,))
        if type(value) is not int or value < 1:
            raise ValueError('prompt lengths must be positive integers, got %r' % (text,))
        targets.append(value)
    if not targets:
        raise ValueError('at least one prompt length is required')
    return targets


def choose_framing(tokenizer, target, task_index, tasks=TASKS, min_excerpt=MIN_EXCERPT_TOKENS):
    """'repository' when make_context_prompt's framing leaves `min_excerpt` tokens of context at
    `target`, else 'compact' (only ever asked in the per-user mode)."""
    overhead = len(encode_prompt(tokenizer, '', tasks[task_index % len(tasks)]))
    return 'repository' if target >= overhead + min_excerpt else 'compact'


class CountingEncoder:
    """encode(excerpt) -> tokens, counting the calls and the seconds they took."""

    def __init__(self, encode):
        self.encode = encode
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, excerpt):
        started = time.perf_counter()
        try:
            return self.encode(excerpt)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - started


def fit_prefix(encode, window, target, *, sample_characters=SAMPLE_CHARACTERS, max_calls=MAX_CALLS):
    """(characters, tokens, template_tokens): a prefix of `window` whose encoded prompt is exactly
    `target` tokens when the search meets one; otherwise the longest prefix found at <= target
    whose next character encodes past it (BPE stepped over the target; pad_to fills the gap).

    The template alone (an empty excerpt) gives the fixed overhead; a sample of the window gives
    characters per token; each step then moves by the local characters per token toward the
    target, inside the bracket (the longest prefix known to fit, the shortest known not to),
    bisecting whenever the secant step would leave it. Refuses a window too small to reach the
    target and a search that has not converged within max_calls tokenizer calls."""
    template = encode('')
    overhead = len(template)
    if overhead >= target:
        raise ValueError('the template alone is %d tokens, no room for a %d-token prompt' % (overhead, target))
    if len(window.encode('utf-8')) < target - overhead:
        raise ValueError('corpus window of %d characters cannot reach %d tokens (a byte-level BPE token covers at '
                         'least one byte, beyond the %d-token template): the corpus is too small for this many users'
                         % (len(window), target, overhead))
    sample = window[:min(len(window), sample_characters)]
    per_token = len(sample) / max(len(encode(sample)) - overhead, 1)
    low, low_tokens = 0, template           # the longest prefix known to encode to <= target
    high = len(window) + 1                  # the shortest known to encode past it (or past the window)
    chars = min(len(window), max(1, int(round((target - overhead) * per_token))))
    calls = 3                               # the template, the sample and the first prefix below
    while True:
        tokens = encode(window[:chars])
        count = len(tokens)
        if count == target:
            return chars, tokens, template
        if count < target:
            low, low_tokens = chars, tokens
            if chars >= len(window):
                raise ValueError('corpus window of %d characters encodes to %d tokens, fewer than %d: the corpus is '
                                 'too small for this many users' % (len(window), count, target))
        else:
            high = chars
        if high - low <= 1:
            return low, low_tokens, template
        if calls >= max_calls:
            raise ValueError('prefix search did not converge in %d tokenizer calls' % max_calls)
        local = chars / max(count - overhead, 1)
        guess = chars + int(round((target - count) * local))
        if not low < guess < high:
            guess = (low + high) // 2
        chars = min(guess, len(window))
        calls += 1


def splice_index(tokens, template):
    """Where the excerpt's tokens end in `tokens`: the start of the longest tail it shares with
    the template (the footer, the task and the generation prompt), never reaching back into the
    head the two share (the system turn and the header)."""
    head, limit = 0, min(len(tokens), len(template))
    while head < limit and tokens[head] == template[head]:
        head += 1
    tail = 0
    while tail < limit - head and tokens[-1 - tail] == template[-1 - tail]:
        tail += 1
    return len(tokens) - tail


def pad_to(tokens, template, target, filler, *, max_padding=MAX_PADDING):
    """(tokens, padding, at): `tokens` made exactly `target` long by `filler` ids spliced in at
    index `at`, just before the footer's tokens; (a copy, 0, None) when it already is."""
    missing = target - len(tokens)
    if missing == 0:
        return list(tokens), 0, None
    if not 0 < missing <= max_padding:
        raise ValueError('no prefix of the window comes within %d tokens of %d (the closest below encodes to %d): '
                         'refusing to pad more' % (max_padding, target, len(tokens)))
    at = splice_index(tokens, template)
    return list(tokens[:at]) + [filler] * missing + list(tokens[at:]), missing, at


def build_prompts(users, target, *, tokenizer=None, model=None, corpus=None, corpus_info=None, tasks=TASKS,
                  max_padding=MAX_PADDING, log=None, targets=None):
    """The real-text prompt set for `users` users of exactly `target` tokens each - or, with
    `targets` (one length per user, len(targets) == users; `target` is then ignored and may be
    None), of exactly targets[i] tokens for user i, short ones in the compact framing.

    `tokenizer` and `corpus` are injectable (CPU tests); in the container they default to
    AutoTokenizer on `model` and the installed vLLM source. Returns a dict with the corpus
    provenance, one entry per user (tokens included) and the build's tokenizer cost."""
    started = time.perf_counter()
    if type(users) is not int or users < 1:
        raise ValueError('users must be a positive integer')
    if targets is not None:
        targets = parse_targets(targets)
        if len(targets) != users:
            raise ValueError('%d prompt lengths for %d users' % (len(targets), users))
    elif type(target) is not int or target < 1:
        raise ValueError('target must be a positive integer')
    if tokenizer is None:
        if model is None:
            raise ValueError('a tokenizer or a model path is required')
        tokenizer = load_tokenizer(model)
    loaded = time.perf_counter()
    if corpus is None:
        corpus, corpus_info = build_corpus(package_root())
    elif corpus_info is None:
        corpus_info = dict(root=None, files=None, characters=len(corpus), sha256=sha256_text(corpus))
    guarded = guarded_tokens(tokenizer)
    cleaned, neutralised = neutralise(corpus, list(guarded))
    guarded_ids = set(guarded.values())
    corpus_built = time.perf_counter()
    entries = []
    filler = None
    for user in range(users):
        start, end = user * len(cleaned) // users, (user + 1) * len(cleaned) // users
        task_index = user % len(tasks)
        framing = 'repository'
        if targets is not None:
            target = targets[user]
            framing = choose_framing(tokenizer, target, task_index, tasks)
        if framing == 'compact':
            task_index = user % len(COMPACT_TASKS)
            encoder = CountingEncoder(lambda excerpt, task=COMPACT_TASKS[task_index]: encode_compact(
                tokenizer, excerpt, task))
        else:
            encoder = CountingEncoder(lambda excerpt, task=tasks[task_index]: encode_prompt(tokenizer, excerpt, task))
        window = cleaned[start:end]
        characters, tokens, template = fit_prefix(encoder, window, target)
        if len(tokens) != target and filler is None:
            filler = filler_token(tokenizer)
        tokens, padding, padding_at = pad_to(tokens, template, target, filler, max_padding=max_padding)
        if len(tokens) != target:
            raise AssertionError('user %d prompt is %d tokens, not %d' % (user, len(tokens), target))
        expected = Counter(token for token in template if token in guarded_ids)
        carried = Counter(token for token in tokens if token in guarded_ids)
        if carried != expected:
            raise ValueError('user %d prompt carries special/added token ids %s, the template alone %s: '
                             'special-token text survived neutralisation' % (user, dict(carried), dict(expected)))
        excerpt = window[:characters]
        if framing == 'compact':
            task_name = COMPACT_TASK_NAMES[task_index]
        else:
            task_name = TASK_NAMES[task_index] if tasks is TASKS else None
        entry = dict(user=user, task_index=task_index, task=task_name,
                     target=target, prompt_tokens=len(tokens), template_tokens=len(template),
                     window_start=start, window_end=end, excerpt_start=start, excerpt_end=start + characters,
                     excerpt_characters=characters, excerpt_sha256=sha256_text(excerpt),
                     prompt_sha256=prompt_sha256(tokens), chunks_of_2048=-(-len(tokens) // 2048),
                     padding_tokens=padding, padding_token_id=filler if padding else None, padding_at=padding_at,
                     tokenizer_calls=encoder.calls, tokenizer_seconds=round(encoder.seconds, 3), tokens=tokens)
        if targets is not None:
            entry['framing'] = framing
        entries.append(entry)
        if log is not None:
            log('[REALTEXT] user=%d task=%s prompt_tokens=%d padding=%d excerpt=[%d,%d) calls=%d seconds=%.2f sha=%s'
                % (user, entry['task'] or task_index, len(tokens), padding, start, start + characters, encoder.calls,
                   encoder.seconds, entry['prompt_sha256'][:16]))
    finished = time.perf_counter()
    info = dict(corpus_info, cleaned_sha256=sha256_text(cleaned), cleaned_characters=len(cleaned),
                neutralised=neutralised)
    built = dict(scope=__doc__.split('\n\n')[0], corpus=info, target=target, max_padding=max_padding, users=entries,
                 system=SYSTEM, header=HEADER, footer=FOOTER, tasks=list(tasks),
                 tokenizer_calls=sum(e['tokenizer_calls'] for e in entries),
                 tokenizer_seconds=round(sum(e['tokenizer_seconds'] for e in entries), 3),
                 seconds=dict(tokenizer_load=round(loaded - started, 3), corpus=round(corpus_built - loaded, 3),
                              prompts=round(finished - corpus_built, 3), total=round(finished - started, 3)))
    if targets is not None:
        built.update(target=None, targets=list(targets), compact_header=COMPACT_HEADER,
                     compact_tasks=list(COMPACT_TASKS))
    return built


def summary(built):
    """The prompt set without the token lists (what the gate's report carries)."""
    result = {key: value for key, value in built.items() if key != 'users'}
    result['users'] = [{key: value for key, value in entry.items() if key != 'tokens'} for entry in built['users']]
    result['prompt_lengths'] = [entry['prompt_tokens'] for entry in built['users']]
    return result


def write_prompts(path, built):
    """The whole set, tokens included, as JSON at `path`; returns the path (None if unwritable)."""
    path = Path(path)
    try:
        path.write_text(json.dumps(built, separators=(',', ':')), encoding='utf-8')
    except OSError:
        return None
    return path
