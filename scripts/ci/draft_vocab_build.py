"""Build the coding draft-vocabulary shortlist (QWEN_FAST_DRAFT_VOCAB, docs/tp4-draft-vocab.md) from PUBLIC text only.

    python3 scripts/ci/draft_vocab_build.py --tokenizer <dir holding tokenizer.json> \\
        --swe <public SWE-rebench parquet> [--swe ...] --code py=<root> --code rs=<root> --code js=<root> --code c=<root> \\
        [--rows 40960] [--name coding-40960] [--out-dir scripts/ci]
    python3 scripts/ci/draft_vocab_build.py --verify scripts/ci/draft_vocab_coding_40960.json

The shortlist restricts which tokens the DFlash2 drafter can PROPOSE. The target still verifies over the full vocabulary, so committed text is
byte-identical whatever this list holds; a token outside it can only cost accepted length (tau), never correctness.

WHAT IT COUNTS. Token frequencies of text shaped like a coding agent's own output (assistant turn: reasoning prose, then a tool call in the
served chat template's `<tool_call><function=..><parameter=..>` markup), rendered deterministically from PUBLIC data and tokenized with the served
tokenizer. No model was run and no customer or own-session text is read:
  edit   the gold patches of the public SWE-rebench dataset (CC-BY-4.0), each hunk rendered as a str_replace / create tool call
  test   the same dataset's test patches, rendered the same way
  prose  its issue statements and maintainer hints, wrapped as thinking (English technical prose)
  shell  bash tool calls built from its test ids, file paths and identifiers
  oss_*  open-source source files found on the building host (python, rust, js/ts, c/c++), sampled by path hash up to a byte budget
Each category is counted on its own, then mixed with WEIGHTS (so a category's size does not decide its weight). Documents are split by a hash of
their repository (SWE-rebench) or path (source files): one in ten is HELD OUT, never counted for selection, and the coverage figures are the
held-out ones. They are a proxy: the shortlist's real cost is the tau lab's `dvocab` arm on the model's own output.

WHAT IT KEEPS UNCONDITIONALLY. Every added/special token of the tokenizer (control, chat, vision, FIM, tool-call, think markers: a marker must stay
proposable) and all 256 single-byte tokens (the byte-level alphabet: any text is spellable). Unused padding rows (ids past the tokenizer's last
token, which the model never emits) are never selected. The rest is the highest mixed frequency, ties to the lower id (BPE merge order).

OUTPUT. `draft_vocab_<name>.ids` (the ascending ids, uint32 little endian, nothing else) and `draft_vocab_<name>.json` (its sha256, the coverage
statistics, the required ids). The rows are a multiple of 128 (four chips x one 32-row tile). No path, host or text of the corpus is written.

Stdlib only for the pure functions (the tests run them); tokenizers, pyarrow and numpy are imported by the command line only.
"""

import argparse
import hashlib
import json
import os
import struct
import sys

FORMAT = 'draft-vocab/1'
VOCAB_SIZE = 248320            # the served LM-head rows (the tokenizer defines fewer; the rest is padding)
ROWS_DEFAULT = 40960
ROW_MULTIPLE = 128             # four chips x one 32-row tile
HELD_OUT_MODULUS = 10          # one document in ten is held out
CURVE_ROWS = (8192, 12288, 16384, 20480, 24576, 32768, 40960, 49152)
COVERAGE_BAR = 0.96

# (category, mixture weight). Code categories are the ones the "generated code tokens" coverage is taken over.
WEIGHTS = (('edit', 0.30), ('test', 0.10), ('prose', 0.25), ('shell', 0.05),
           ('oss_py', 0.10), ('oss_rs', 0.06), ('oss_js', 0.06), ('oss_c', 0.08))
CODE_CATEGORIES = ('edit', 'test', 'oss_py', 'oss_rs', 'oss_js', 'oss_c')
LANGUAGES = {
    'py': ('oss_py', ('.py',)),
    'rs': ('oss_rs', ('.rs',)),
    'js': ('oss_js', ('.js', '.mjs', '.cjs', '.ts', '.tsx', '.jsx')),
    'c': ('oss_c', ('.c', '.h', '.cc', '.cpp', '.hpp', '.hh', '.cu', '.cuh')),
}
FILE_BYTES = (200, 200000)     # a source file outside this size is skipped (stubs, generated bundles)
MAX_LINE = 500                 # a file with a longer line is minified or generated: skipped
DOCUMENT_CHARS = 24000         # one rendered document is cut here


# ---- the pure parts -------------------------------------------------------------------------------------------------------------------

def bytes_to_unicode():
    """GPT-2's byte -> printable character map, the byte-level BPE alphabet: {byte: character}."""
    kept = list(range(ord('!'), ord('~') + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    mapping, extra = {}, 0
    for byte in range(256):
        if byte in kept:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + extra)
            extra += 1
    return mapping


def byte_token_ids(vocabulary):
    """The ids of the 256 single-byte tokens in `vocabulary` ({token string: id}); a missing one is an error (the alphabet is the fallback)."""
    ids = []
    for byte, character in sorted(bytes_to_unicode().items()):
        if character not in vocabulary:
            raise ValueError('the tokenizer has no single-byte token for byte %d' % byte)
        ids.append(vocabulary[character])
    return tuple(sorted(ids))


def special_token_ids(added_tokens):
    """Every added token of the tokenizer (special or not: <tool_call>, <think> and the FIM markers are added tokens without the special flag)."""
    return tuple(sorted(int(token['id']) for token in added_tokens))


def required_from_tokenizer(tokenizer):
    """(added ids, single-byte ids, the number of real tokens) from a parsed tokenizer.json."""
    vocabulary = tokenizer['model']['vocab']
    added = special_token_ids(tokenizer['added_tokens'])
    real = max(max(vocabulary.values()), max(added)) + 1
    return added, byte_token_ids(vocabulary), real


def split_of(key):
    """'heldout' for one key in HELD_OUT_MODULUS (a repository name or a relative path), else 'train'. Stable across runs and hosts."""
    digest = hashlib.sha256(key.encode('utf-8', 'surrogatepass')).digest()
    return 'heldout' if int.from_bytes(digest[:4], 'big') % HELD_OUT_MODULUS == 0 else 'train'


def mixture(counts, weights=WEIGHTS):
    """{token: mixed frequency} from {category: {token: count}}: sum over present categories of weight x count / category total, the weights renormalised
    over the categories that have tokens at all. -> (mixed, the weights actually used)."""
    present = [(name, weight) for name, weight in weights if sum(counts.get(name, {}).values()) > 0]
    if not present:
        raise ValueError('no category has any token')
    total = float(sum(weight for _, weight in present))
    used = {name: weight / total for name, weight in present}
    mixed = {}
    for name, weight in used.items():
        column = counts[name]
        size = float(sum(column.values()))
        for token, count in column.items():
            if count:
                mixed[token] = mixed.get(token, 0.0) + weight * count / size
    return mixed, used


def select(mixed, required, rows, real_tokens):
    """The shortlist: `required` plus the highest mixed frequencies (ties to the lower id) up to `rows` ids, ascending. Padding ids (>= real_tokens) are never
    chosen; if the counted tokens run out, the lowest unused real ids fill the rest (BPE order is roughly frequency order)."""
    required = tuple(sorted(set(required)))
    if (type(rows) is not int or rows % ROW_MULTIPLE or not 0 < rows <= real_tokens
            or any(not 0 <= token < real_tokens for token in required)):
        raise ValueError('rows must be a positive multiple of %d within the real tokens, required ids real' % ROW_MULTIPLE)
    if len(required) > rows:
        raise ValueError('the required tokens (%d) do not fit in %d rows' % (len(required), rows))
    chosen = set(required)
    for token in sorted((token for token in mixed if token < real_tokens), key=lambda token: (-mixed[token], token)):
        if len(chosen) == rows:
            break
        chosen.add(token)
    token = 0
    while len(chosen) < rows:
        chosen.add(token)
        token += 1
    return tuple(sorted(chosen))


def coverage(column, chosen):
    """The fraction of the occurrences in {token: count} whose token is in `chosen` (a set); None for an empty column."""
    total = sum(column.values())
    if not total:
        return None
    return sum(count for token, count in column.items() if token in chosen) / float(total)


def code_coverage(held_out, chosen, used):
    """(weighted, worst, {category: coverage}) over the held-out code categories that have tokens; `used` are the mixture weights."""
    per = {}
    for name in CODE_CATEGORIES:
        value = coverage(held_out.get(name, {}), chosen)
        if value is not None:
            per[name] = value
    if not per:
        return None, None, per
    weight = sum(used.get(name, 0.0) for name in per)
    weighted = sum(used.get(name, 0.0) * value for name, value in per.items()) / weight if weight else sum(per.values()) / len(per)
    return weighted, min(per.values()), per


def pack_ids(ids):
    """uint32 little endian, nothing else (160 KiB for 40,960 rows)."""
    return b''.join(struct.pack('<I', token) for token in ids)


def unpack_ids(data):
    if len(data) % 4:
        raise ValueError('an id list is a whole number of uint32 words')
    return tuple(struct.unpack('<%dI' % (len(data) // 4), data))


def sha256_of(data):
    return hashlib.sha256(data).hexdigest()


def hunks(patch):
    """[(path, is_new_file, old text, new text)] one per hunk of a unified diff (context lines are in both sides)."""
    found, path, new_file, old, new, inside = [], None, False, [], [], False

    def flush():
        if path is not None and (old or new):
            found.append((path, new_file, '\n'.join(old), '\n'.join(new)))

    for line in patch.split('\n'):
        if line.startswith('diff --git '):
            flush()
            old, new, inside, new_file = [], [], False, False
            parts = line.split(' b/', 1)
            path = parts[1] if len(parts) == 2 else None
        elif line.startswith('new file mode'):
            new_file = True
        elif line.startswith('@@'):
            flush()
            old, new, inside = [], [], True
        elif inside:
            if line.startswith('+') and not line.startswith('+++'):
                new.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                old.append(line[1:])
            elif line.startswith(' '):
                old.append(line[1:])
                new.append(line[1:])
    flush()
    return found


def tool_call(name, parameters):
    """The served chat template's tool-call markup for one call."""
    body = ''.join('<parameter=%s>\n%s\n</parameter>\n' % (key, value) for key, value in parameters)
    return '<tool_call>\n<function=%s>\n%s</function>\n</tool_call>' % (name, body)


def render_edits(patch):
    """The patch as str_replace / create tool calls (the model's edit output), one document per file so a long patch is not cut mid-call."""
    documents = {}
    for path, new_file, old, new in hunks(patch):
        if new_file:
            call = tool_call('str_replace_editor', [('command', 'create'), ('path', '/testbed/' + path), ('file_text', new)])
        else:
            call = tool_call('str_replace_editor', [('command', 'str_replace'), ('path', '/testbed/' + path),
                                                    ('old_str', old), ('new_str', new)])
        documents[path] = documents.get(path, '') + call + '\n'
    return [text[:DOCUMENT_CHARS] for text in documents.values()]


def render_prose(statement, hints):
    """Issue text and maintainer hints as the model's thinking (English technical prose around code)."""
    documents = []
    for text in (statement, hints):
        text = (text or '').strip()
        if len(text) >= 200:
            documents.append('<think>\n' + text[:DOCUMENT_CHARS] + '\n</think>\n\n')
    return documents


def identifiers(patch):
    """Function and class names the patch touches, in order of appearance, unique."""
    found = []
    for line in patch.split('\n'):
        body = line[1:].lstrip() if line[:1] in ('+', '-', ' ') else ''
        for keyword in ('def ', 'class '):
            if body.startswith(keyword):
                name = body[len(keyword):].split('(')[0].split(':')[0].strip()
                if name and name.isidentifier() and name not in found:
                    found.append(name)
    return found


def render_shell(patch, tests):
    """A few bash tool calls the agent would make around the patch: run the failing tests, grep the touched names, read and diff the files."""
    files = [path for path, unused, unused2, unused3 in hunks(patch)]
    files = list(dict.fromkeys(files))
    names = identifiers(patch)
    commands = ['cd /testbed && python -m pytest %s -x -q 2>&1 | tail -30' % test for test in list(tests or [])[:2]]
    commands += ['grep -rn "%s" %s/' % (name, os.path.dirname(files[0]) or '.') for name in names[:2]] if files else []
    commands += ["sed -n '1,120p' %s" % path for path in files[:2]]
    commands += ['git diff -- %s' % path for path in files[:2]]
    commands += ['find . -name "*.py" | xargs grep -ln "%s"' % name for name in names[:1]]
    return [tool_call('bash', [('command', command)]) + '\n' for command in commands]


def verify(sidecar_path):
    """Problems with a committed list and its sidecar, [] when sound: sha256, ordering, bounds, the required ids, the recorded coverage."""
    problems = []
    with open(sidecar_path, encoding='utf-8') as handle:
        meta = json.load(handle)
    ids_path = os.path.join(os.path.dirname(os.path.abspath(sidecar_path)), meta['ids_file'])
    with open(ids_path, 'rb') as handle:
        data = handle.read()
    ids = unpack_ids(data)
    if sha256_of(data) != meta['ids_sha256']:
        problems.append('ids_sha256 does not match the list')
    if len(ids) != meta['rows'] or len(ids) % ROW_MULTIPLE:
        problems.append('rows %d is not the recorded %d or not a multiple of %d' % (len(ids), meta['rows'], ROW_MULTIPLE))
    if any(a >= b for a, b in zip(ids, ids[1:])) or (ids and (ids[0] < 0 or ids[-1] >= meta['real_tokens'])):
        problems.append('ids are not strictly ascending real tokens')
    have = set(ids)
    for kind in ('added_tokens', 'single_byte_tokens'):
        missing = [token for token in meta['required'][kind] if token not in have]
        if missing:
            problems.append('%d required %s missing' % (len(missing), kind))
    bar = meta['coverage']['bar']
    if meta['coverage']['heldout_code_weighted'] < bar:
        problems.append('held-out weighted code coverage %.4f is under the %.2f bar' % (meta['coverage']['heldout_code_weighted'], bar))
    return problems


# ---- the command line (needs tokenizers, pyarrow, numpy) ------------------------------------------------------------------------------

def swe_documents(paths):
    """(category, split, text) for every rendered document of the public SWE-rebench parquet files, plus {split: instances}."""
    import pyarrow.parquet as parquet

    columns = ['repo', 'patch', 'test_patch', 'problem_statement', 'hints_text', 'FAIL_TO_PASS']
    seen, instances = set(), {'train': 0, 'heldout': 0}
    for path in paths:
        handle = parquet.ParquetFile(path)
        for group in range(handle.num_row_groups):
            for row in handle.read_row_group(group, columns=columns).to_pylist():
                key = (row['repo'], row['problem_statement'][:200])
                if key in seen:
                    continue
                seen.add(key)
                split = split_of(row['repo'])
                instances[split] += 1
                for text in render_edits(row['patch'] or ''):
                    yield 'edit', split, text
                for text in render_edits(row['test_patch'] or ''):
                    yield 'test', split, text
                for text in render_prose(row['problem_statement'], row['hints_text']):
                    yield 'prose', split, text
                for text in render_shell(row['patch'] or '', row['FAIL_TO_PASS']):
                    yield 'shell', split, text
    swe_documents.instances = instances


def source_documents(language, root, budget_bytes):
    """(category, split, text) for hashed-path-sampled source files under `root` up to `budget_bytes`, plus the per-split file counts in .files."""
    category, extensions = LANGUAGES[language]
    candidates = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in ('.git', '__pycache__'))
        for name in sorted(files):
            if name.endswith(extensions) and not name.endswith(('.min.js', '.d.ts')):
                path = os.path.join(directory, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if FILE_BYTES[0] <= size <= FILE_BYTES[1]:
                    relative = os.path.relpath(path, root)
                    candidates.append((hashlib.sha256(relative.encode('utf-8', 'surrogatepass')).hexdigest(), relative, path))
    candidates.sort()
    used, files = 0, {'train': 0, 'heldout': 0}
    for _, relative, path in candidates:
        if used >= budget_bytes:
            break
        try:
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        if max((len(line) for line in text.split('\n')), default=0) > MAX_LINE:
            continue
        used += len(text)
        split = split_of(relative)
        files[split] += 1
        yield category, split, text[:DOCUMENT_CHARS]
    source_documents.files = files
    source_documents.bytes = used


def count_tokens(tokenizer, documents, counts, sizes, vocabulary_size):
    """Add the documents' token counts to counts[(category, split)] (a numpy array each) and sizes[(category, split)] = [documents, characters]."""
    import numpy

    batch = {}

    def flush():
        for (category, split), texts in batch.items():
            column = counts.setdefault((category, split), numpy.zeros(vocabulary_size, dtype=numpy.int64))
            for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
                column += numpy.bincount(numpy.asarray(encoding.ids, dtype=numpy.int64), minlength=vocabulary_size)
            texts.clear()

    pending = 0
    for category, split, text in documents:
        batch.setdefault((category, split), []).append(text)
        size = sizes.setdefault((category, split), [0, 0])
        size[0] += 1
        size[1] += len(text)
        pending += 1
        if pending >= 512:
            flush()
            pending = 0
    flush()


def column_of(array):
    return {int(token): int(array[token]) for token in array.nonzero()[0]}


def build(options):
    from tokenizers import Tokenizer

    directory = options.tokenizer
    with open(os.path.join(directory, 'tokenizer.json'), encoding='utf-8') as handle:
        raw = json.load(handle)
    added, single, real = required_from_tokenizer(raw)
    tokenizer = Tokenizer.from_file(os.path.join(directory, 'tokenizer.json'))
    counts, sizes = {}, {}
    count_tokens(tokenizer, swe_documents(options.swe), counts, sizes, VOCAB_SIZE)
    instances = dict(swe_documents.instances)
    files = {}
    for spec in options.code:
        language, root = spec.split('=', 1)
        count_tokens(tokenizer, source_documents(language, root, int(options.budget_mb * 1e6)), counts, sizes, VOCAB_SIZE)
        old = files.get(language, {'train': 0, 'heldout': 0})
        files[language] = {split: old[split] + source_documents.files[split] for split in old}
    train = {name: column_of(counts[(name, 'train')]) for name, _ in WEIGHTS if (name, 'train') in counts}
    held = {name: column_of(counts[(name, 'heldout')]) for name, _ in WEIGHTS if (name, 'heldout') in counts}
    mixed, used = mixture(train)
    required = set(added) | set(single)
    ids = select(mixed, required, options.rows, real)
    chosen = set(ids)
    weighted, worst, per = code_coverage(held, chosen, used)
    curve = {}
    for rows in CURVE_ROWS:
        if rows <= real:
            other = set(select(mixed, required, rows, real))
            value, low, unused = code_coverage(held, other, used)
            overall = {name: coverage(column, other) for name, column in held.items()}
            curve[str(rows)] = dict(code_weighted=round(value, 5), code_worst=round(low, 5),
                                    all_categories=round(sum(used[name] * v for name, v in overall.items() if v is not None and name in used) /
                                                         sum(used[name] for name, v in overall.items() if v is not None and name in used), 5))
    data = pack_ids(ids)
    stem = 'draft_vocab_' + options.name.replace('-', '_')
    meta = dict(
        format=FORMAT, name=options.name, rows=len(ids), vocab_size=VOCAB_SIZE, real_tokens=real, ids_file=stem + '.ids',
        ids_sha256=sha256_of(data), ids_bytes=len(data),
        required=dict(added_tokens=list(added), single_byte_tokens=list(single)),
        selection=dict(weights={name: round(weight, 4) for name, weight in used.items()}, tie_break='lower token id',
                       required_count=len(required), by_frequency=len(ids) - len(required & chosen),
                       held_out='one document in %d by repository / path hash' % HELD_OUT_MODULUS),
        corpus=dict(
            swe_rebench=dict(source='public dataset nebius/SWE-rebench (CC-BY-4.0)', instances=instances),
            open_source_files=files,
            documents={'%s/%s' % key: dict(documents=value[0], characters=value[1]) for key, value in sorted(sizes.items())},
            tokens={'%s/%s' % key: int(counts[key].sum()) for key in sorted(counts)}),
        coverage=dict(bar=COVERAGE_BAR, heldout_code_weighted=round(weighted, 5), heldout_code_worst_category=round(worst, 5),
                      heldout_by_category={name: round(value, 5) for name, value in sorted(per.items())},
                      heldout_prose=round(coverage(held.get('prose', {}), chosen) or 0.0, 5),
                      heldout_shell=round(coverage(held.get('shell', {}), chosen) or 0.0, 5),
                      train_code_weighted=round(code_coverage(train, chosen, used)[0], 5),
                      curve_by_rows=curve),
        notes='A proxy, not the model: no model was run. The coverage figures are on held-out public documents; the real cost of the shortlist is the accepted '
              'length it removes, measured by the tau lab arm dvocab on the model\'s own output. The target verifies the full vocabulary either way.')
    os.makedirs(options.out_dir, exist_ok=True)
    with open(os.path.join(options.out_dir, stem + '.ids'), 'wb') as handle:
        handle.write(data)
    with open(os.path.join(options.out_dir, stem + '.json'), 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    print('rows=%d sha256=%s heldout_code_weighted=%.4f worst=%.4f' % (len(ids), meta['ids_sha256'], weighted, worst))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tokenizer', help='a directory holding the served tokenizer.json')
    parser.add_argument('--swe', action='append', default=[], help='a public SWE-rebench parquet file (repeatable)')
    parser.add_argument('--code', action='append', default=[], help='LANG=ROOT with LANG one of %s (repeatable)' % ', '.join(sorted(LANGUAGES)))
    parser.add_argument('--budget-mb', type=float, default=12.0, help='source text per --code root, in MB')
    parser.add_argument('--rows', type=int, default=ROWS_DEFAULT)
    parser.add_argument('--name', default='coding-40960')
    parser.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--verify', help='check a committed sidecar and its list, exit 1 on a problem')
    options = parser.parse_args(argv)
    if options.verify:
        problems = verify(options.verify)
        for problem in problems:
            sys.stderr.write(problem + '\n')
        return 1 if problems else 0
    if not options.tokenizer or not options.swe:
        parser.error('--tokenizer and at least one --swe file are required to build')
    for spec in options.code:
        if spec.split('=', 1)[0] not in LANGUAGES or '=' not in spec:
            parser.error('--code takes LANG=ROOT with LANG in %s' % ', '.join(sorted(LANGUAGES)))
    build(options)
    return 0


if __name__ == '__main__':
    sys.exit(main())
