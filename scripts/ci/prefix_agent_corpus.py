"""A real-text multi-turn coding-agent corpus for the prefix-reuse gates (TT prefix-reuse design 2.2).

The G1 gates need conversations shaped like the traffic prefix reuse is for - a coding agent with
tools, reasoning on, six or more turns, 4k to 60k tokens of context - built from REAL text. Synthetic
prompts hide context bugs (memory synthetic-prompt-hides-context-bugs), and the platform gateway
drops request content by default, so there is no captured traffic to replay: this module writes it.

What a conversation is here:
  - a system block (SYSTEM_KINDS): 'full' is an OpenCode / Claude-Code style agent prompt, a fixed
    environment block, a real README as project instructions and nine tool schemas (about 6k tokens,
    the P0b study's shape); 'compact' is a short agent prompt with three tools (about 1k tokens), so
    a first turn can sit below 2048 tokens and the boundary cases (2047/2048/2049) can be built;
  - a first user message: a coding task naming a real file and a real function from the corpus,
    optionally with a real excerpt attached;
  - then, turn after turn, the MODEL'S OWN answer (the replay driver appends what the server
    returned: content, reasoning, tool calls) followed by the next input: one tool result per tool
    call (real file excerpts in the read tool's `cat -n` shape, real grep lines, real paths) or, when
    the answer made no tool call, a user follow-up with a real excerpt pasted in.
Each input is sized to a token budget the driver computes from the served token counts (a growth
plan to a target length, or the metering shape: ~2k-token tool results, exponential think gaps).

Everything is deterministic in (seed, conversation name, turn, variant): the same conversation state
always gets the same next input, so a cold arm and a hit arm are sent byte-identical messages.

Real text comes from files on disk (load_sources): on the rig, the checkout's own scripts/ci, docs,
docker and workflow files. corpus_info records what was used (file count, characters, sha256).

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import copy
import fnmatch
import hashlib
import json
import os
import random
import re

CHARS_PER_TOKEN = 3.2          # Qwen3.8's tokenizer on this repo's numbered excerpts: 3.23 measured (a sizing estimate)
MIN_INPUT_CHARS = 160
MIN_LINE_CHARS = 16            # a line cut to its budget keeps at least its number and a few characters
SOURCE_DIRS = ('scripts/ci', 'docs', 'docker', '.github/workflows')
SOURCE_EXTENSIONS = ('.py', '.md', '.sh', '.yml', '.yaml', '.txt', '.cpp', '.h', '.toml', '.json')
MIN_FILE_CHARS = 400
MAX_FILE_CHARS = 400000
MAX_PROMPT_TOKENS = 60000      # the general profile's 65536 less a long answer
README_CHARS = 9000
# An excerpt is file blocks until its budget is met; the bound only stops a corpus of tiny files
# from looping (the tiny arm's filler asks ~60k tokens of one message).
MAX_EXCERPT_BLOCKS = 4096

# The metering shape (SKILL "Real traffic to size against"; design SIM): tool results average ~2k
# tokens, answers ~740 tokens, think gaps average 15 s; a conversation compacts past ~60k.
METERING = dict(input_tokens_mean=2000, input_tokens_min=300, input_tokens_max=6000, gap_mean_s=15.0,
                max_prompt_tokens=MAX_PROMPT_TOKENS, compaction_tokens=2000)


class Source(object):
    __slots__ = ('path', 'text', 'lines')

    def __init__(self, path, text):
        self.path = path
        self.text = text
        self.lines = text.split('\n')
        if self.lines and self.lines[-1] == '':
            self.lines.pop()


class FitError(ValueError):
    """A prompt could not be built to the exact token count asked."""


def load_sources(root, dirs=SOURCE_DIRS, extensions=SOURCE_EXTENSIONS, limit=None):
    """Every readable UTF-8 text file under root/<dir> with one of `extensions`, between
    MIN_FILE_CHARS and MAX_FILE_CHARS long, CRLF normalised, sorted by repository path."""
    found = []
    for directory in dirs:
        base = os.path.join(root, directory)
        for current, subdirs, files in os.walk(base):
            subdirs[:] = sorted(name for name in subdirs if not name.startswith('.') and name != '__pycache__')
            for name in sorted(files):
                if not name.endswith(tuple(extensions)):
                    continue
                path = os.path.join(current, name)
                try:
                    with open(path, 'rb') as handle:
                        raw = handle.read(MAX_FILE_CHARS + 1)
                    text = raw.decode('utf-8')
                except (OSError, UnicodeDecodeError):
                    continue
                text = text.replace('\r\n', '\n')
                if MIN_FILE_CHARS <= len(text) <= MAX_FILE_CHARS and '\x00' not in text:
                    found.append(Source(os.path.relpath(path, root).replace(os.sep, '/'), text))
    found.sort(key=lambda source: source.path)
    return found[:limit] if limit else found


def corpus_info(sources):
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.path.encode('utf-8') + b'\0' + source.text.encode('utf-8') + b'\0')
    return dict(files=len(sources), characters=sum(len(source.text) for source in sources),
                sha256=digest.hexdigest())


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN)


def token_chars(tokens):
    return max(MIN_INPUT_CHARS, int(tokens * CHARS_PER_TOKEN))


def numbered(lines, start, limit_chars):
    """`cat -n` lines (the read tool's shape) from index `start`, up to limit_chars characters. The
    first line always comes, but cut to the budget when it alone is longer (a read tool truncates a
    long line too): G1 v48 (run 36251045616) sent a 2,500-token tool result whose excerpt began on
    scripts/ci/frozen-ladder-corpus.json's single 322,263-character line, and the prompt passed the
    65,536-token context at the API edge."""
    out, used = [], 0
    for index in range(start, len(lines)):
        line = '%6d\t%s' % (index + 1, lines[index])
        if used + len(line) + 1 > limit_chars:
            if out:
                break
            line = line[:max(MIN_LINE_CHARS, int(limit_chars) - 1)]
        out.append(line)
        used += len(line) + 1
    return '\n'.join(out)


# -- system blocks and tools ---------------------------------------------------------------------

AGENT_PROMPT_FULL = """You are an interactive coding agent running in the user's terminal. You help with software engineering tasks: fixing bugs, adding features, refactoring, explaining code, and running and repairing tests. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Refuse to write or improve code that is plainly intended to be malicious. Do not guess URLs; only use URLs the user gave you or that appear in local files.

# Tone and style
- Be concise, direct and to the point. Your output is shown in a monospace terminal, rendered as GitHub-flavoured Markdown.
- Answer in fewer than four lines unless the user asks for detail. Do not add preamble or postamble such as "Here is what I did" unless asked.
- When you run a non-trivial command, explain what it does and why, especially if it changes the user's system.
- Communicate with the user through your text output only. Never use tools such as bash echo or code comments to talk to the user.

# Proactiveness
You may act proactively, but only when the user asks you to do something. Strike a balance between doing the right thing when asked, including follow-up actions, and not surprising the user with actions they did not ask for. If the user asks how to approach something, answer the question first and do not immediately start editing files.

# Following conventions
When you change files, first understand the file's conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- Never assume a library is available, even a well-known one. Check that the codebase already uses it: look at neighbouring files, or the package manifest.
- When you edit code, look at the surrounding context, especially its imports, to understand the choice of frameworks and libraries. Then make the change in the most idiomatic way.
- Follow security best practice. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.

# Task management
You have the todowrite tool to plan and track tasks. Use it frequently, so that you track your tasks and give the user visibility into your progress. Mark todos as completed as soon as you finish a task.

# Doing tasks
- Use the search tools to understand the codebase and the user's query. Use them extensively, in parallel and in sequence.
- Implement the solution using all tools available to you.
- Verify the solution with tests if possible. Never assume a particular test framework or test script. Check the README or search the codebase to find the testing approach.
- NEVER commit changes unless the user explicitly asks you to.

# Tool usage policy
- You can call several tools in a single response. When you request multiple independent pieces of information, batch your tool calls together for optimal performance.
- Use the read tool to read files, not cat, head or tail. Use the edit tool to edit files, not sed or awk. Use the grep and glob tools to search, not find or grep in bash.

# Code references
When referencing specific functions or pieces of code, use the pattern `file_path:line_number` so that the user can navigate to the source."""

AGENT_PROMPT_COMPACT = """You are a coding agent working in the user's repository through tools. Read before you edit, keep changes minimal and in the file's existing style, cite code as file_path:line_number, and verify with the project's own tests when you can. Be concise: your output is shown in a terminal."""

# Fixed on purpose: a line that changes per request (a date, a clock) breaks the prefix at that line
# (P0b: a changed date line diverges at 4,223 tokens).
ENVIRONMENT = """Here is useful information about the environment you are running in:
<env>
Working directory: /home/dev/work/Tenstorrent.Blackhole-Qwen3.8-27B
Is directory a git repo: yes
Platform: linux
OS Version: Linux 6.8.0-45-generic
Today's date: 2026-09-26
</env>
You are powered by the model Qwen3.8-27B."""


def _tool(name, description, properties, required):
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required,
                       'additionalProperties': False}}}


_READ = _tool('read', 'Reads a file from the local filesystem. The file_path parameter must be an absolute path. By '
              'default it reads up to 2000 lines from the start of the file; offset and limit select a range. '
              'Results are returned in cat -n format, with line numbers starting at 1.',
              {'file_path': {'type': 'string', 'description': 'The absolute path to the file to read'},
               'offset': {'type': 'integer', 'description': 'The line number to start reading from'},
               'limit': {'type': 'integer', 'description': 'The number of lines to read'}}, ['file_path'])
_GREP = _tool('grep', 'A search tool built on ripgrep. Supports full regex syntax. Filter files with the glob '
              'parameter. Output modes: "content" shows matching lines, "files_with_matches" only paths.',
              {'pattern': {'type': 'string', 'description': 'The regular expression to search for'},
               'path': {'type': 'string', 'description': 'File or directory to search in'},
               'glob': {'type': 'string', 'description': 'Glob pattern to filter files'},
               'output_mode': {'type': 'string', 'enum': ['content', 'files_with_matches', 'count']}}, ['pattern'])
_BASH = _tool('bash', 'Executes a bash command in a persistent shell session with an optional timeout. Quote '
              'paths that contain spaces. Output longer than 30000 characters is truncated.',
              {'command': {'type': 'string', 'description': 'The command to execute'},
               'timeout': {'type': 'integer', 'description': 'Optional timeout in milliseconds'},
               'description': {'type': 'string', 'description': 'What the command does in 5-10 words'}},
              ['command', 'description'])
_EDIT = _tool('edit', 'Performs exact string replacements in files. You must read a file before editing it. The '
              'edit fails if old_string is not found, or is found more than once without replace_all.',
              {'file_path': {'type': 'string', 'description': 'The absolute path to the file to modify'},
               'old_string': {'type': 'string', 'description': 'The text to replace'},
               'new_string': {'type': 'string', 'description': 'The text to replace it with'},
               'replace_all': {'type': 'boolean', 'description': 'Replace every occurrence'}},
              ['file_path', 'old_string', 'new_string'])
_WRITE = _tool('write', 'Writes a file to the local filesystem, overwriting any existing file. Read an existing '
               'file first. Never create documentation files unless asked.',
               {'file_path': {'type': 'string', 'description': 'The absolute path to the file to write'},
                'content': {'type': 'string', 'description': 'The content to write to the file'}},
               ['file_path', 'content'])
_GLOB = _tool('glob', 'Fast file pattern matching. Supports patterns like "**/*.py". Returns matching paths '
              'sorted by modification time.',
              {'pattern': {'type': 'string', 'description': 'The glob pattern to match files against'},
               'path': {'type': 'string', 'description': 'The directory to search in'}}, ['pattern'])
_TODO = _tool('todowrite', 'Create and manage a structured task list for the current session: mark a task '
              'in_progress before starting it and completed as soon as it is done.',
              {'todos': {'type': 'array', 'items': {'type': 'object', 'properties': {
                  'content': {'type': 'string'}, 'status': {'type': 'string',
                                                            'enum': ['pending', 'in_progress', 'completed']},
                  'id': {'type': 'string'}}, 'required': ['content', 'status', 'id']}}}, ['todos'])
_TASK = _tool('task', 'Launch a read-only sub-agent for a multi-step search. It returns one report message.',
              {'description': {'type': 'string', 'description': 'A short (3-5 words) description'},
               'prompt': {'type': 'string', 'description': 'The task for the agent to perform'}},
              ['description', 'prompt'])
_WEBFETCH = _tool('webfetch', 'Fetches content from a URL and returns it as markdown. Read-only.',
                  {'url': {'type': 'string', 'description': 'The URL to fetch'}}, ['url'])

SYSTEM_KINDS = ('full', 'compact')
# A first prompt's tokens with no attachment (system block, tools, a task), measured with Qwen3.8's
# tokenizer and chat template on this repository (2026-09-26): 4,270 full, 947 compact.
SYSTEM_TOKENS = dict(full=4300, compact=1000)
TOOLS = dict(full=[_BASH, _READ, _EDIT, _WRITE, _GREP, _GLOB, _TODO, _TASK, _WEBFETCH],
             compact=[_READ, _GREP, _BASH])

TASKS = (
    'Read {path} and explain what `{name}` does and where it is called from. Then propose one unit test for it.',
    'There is a bug report against `{name}` in {path}: it misbehaves when its input is empty. Find the cause and '
    'fix it with the smallest change.',
    'Refactor `{name}` in {path} so that it is easier to test, without changing its behaviour. Run the tests '
    'afterwards.',
    'Add a docstring to `{name}` in {path} that states its inputs, outputs and failure modes, and check that '
    'every caller passes the right arguments.',
    'Review {path} for error handling around `{name}`: list each place an exception can escape, then fix the '
    'worst one.',
    'Find every place that calls `{name}` (defined in {path}) and summarise how the callers use what it returns.',
)
FOLLOWUPS = (
    'Good. Now check the callers of that code as well, and show me the relevant lines.',
    'That is not quite right - look at this part of the code again and try again:',
    'Before you change anything else, explain your plan in three bullet points. For reference:',
    'Please also handle the edge case where the input is None. Here is related code:',
    'Keep going. Here is more context from a related file:',
    'The tests fail with the change. This is the code the failing test exercises:',
)
COMPACTION = ('This session is being continued from a previous conversation that ran out of context. The '
              'conversation is summarized below, followed by the files that were open.\n\n')
NAME = re.compile(r'^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)', re.M)


class Corpus(object):
    """Real text to build agent traffic from: sources on disk, the system blocks, and the inputs a
    turn gets (tool results, user follow-ups) at a size budget. Pure: no network, no clock."""

    def __init__(self, sources):
        if not sources:
            raise ValueError('the corpus has no sources')
        self.sources = list(sources)
        self.by_name = {}
        for source in self.sources:
            self.by_name.setdefault(os.path.basename(source.path), []).append(source)
        self.info = corpus_info(self.sources)
        readmes = [s for s in self.sources if os.path.basename(s.path).lower() == 'readme.md']
        self.readme = (readmes[0] if readmes else self.sources[0]).text[:README_CHARS]

    # -- system ---------------------------------------------------------------------------------
    def system_text(self, kind):
        if kind == 'full':
            return (AGENT_PROMPT_FULL + '\n\n' + ENVIRONMENT + '\n\n# Project instructions (AGENTS.md)\n\n'
                    + self.readme.strip())
        if kind == 'compact':
            return AGENT_PROMPT_COMPACT + '\n\n' + ENVIRONMENT
        raise ValueError('unknown system kind %r (known: %s)' % (kind, ', '.join(SYSTEM_KINDS)))

    @staticmethod
    def tools(kind):
        return copy.deepcopy(TOOLS[kind])

    # -- real text ------------------------------------------------------------------------------
    def pick(self, rng, hint=None, suffixes=None):
        """A source: the one a tool call names (by basename) when it exists, else a seeded choice."""
        if hint:
            named = self.by_name.get(os.path.basename(str(hint).rstrip('/')))
            if named:
                return named[0]
        pool = self.sources
        if suffixes:
            pool = [source for source in self.sources if source.path.endswith(tuple(suffixes))] or self.sources
        return pool[rng.randrange(len(pool))]

    def excerpt(self, rng, chars, hint=None, offset=None):
        """Numbered lines of real files, about `chars` long: the named (or a seeded) file from a
        seeded line, continued into further files, each headed by its path, until the budget is met."""
        chars = max(MIN_INPUT_CHARS, int(chars))
        parts, used, source, first = [], 0, self.pick(rng, hint), True
        guard = 0
        while used < chars and guard < MAX_EXCERPT_BLOCKS:
            guard += 1
            if first and offset is not None:
                start = max(0, min(len(source.lines) - 1, int(offset) - 1))
            else:
                start = rng.randrange(max(1, len(source.lines) - 20)) if len(source.lines) > 40 else 0
            block = numbered(source.lines, start, chars - used)
            if not block:
                block = numbered(source.lines, start, max(MIN_LINE_CHARS, chars - used))
            text = block if first else '\n==> %s <==\n%s' % (source.path, block)
            parts.append(text)
            used += len(text) + 1
            first = False
            source = self.sources[rng.randrange(len(self.sources))]
        return '\n'.join(parts)

    def function_name(self, rng, source):
        names = NAME.findall(source.text)
        if names:
            return names[rng.randrange(len(names))]
        words = re.findall(r'[A-Za-z_]{6,}', source.text)
        return words[rng.randrange(len(words))] if words else 'main'

    def task(self, rng, chars=0):
        """The first user message: a coding task on a real file and function, with a real excerpt of
        about `chars` characters attached when chars > 0."""
        source = self.pick(rng, suffixes=('.py',))
        text = TASKS[rng.randrange(len(TASKS))].format(path=source.path, name=self.function_name(rng, source))
        if chars > 0:
            text += '\n\nHere is the part of %s I am looking at:\n```\n%s\n```' % (
                source.path, self.excerpt(rng, chars, hint=source.path))
        return text

    def grep(self, rng, pattern, chars):
        """Real `path:line:text` lines that contain the pattern (literal, case-insensitive), padded
        with the first matching file's context when the matches fall short of the budget."""
        needle = re.sub(r'[\\^$.*+?()\[\]{}|]', '', str(pattern or '')).strip().lower()[:40]
        out, used, first = [], 0, None
        if needle:
            start = rng.randrange(len(self.sources))
            for step in range(len(self.sources)):
                source = self.sources[(start + step) % len(self.sources)]
                for index, line in enumerate(source.lines):
                    if needle in line.lower():
                        entry = '%s:%d:%s' % (source.path, index + 1, line[:300])
                        out.append(entry)
                        used += len(entry) + 1
                        first = first or source
                        if used >= chars:
                            return '\n'.join(out)
                if used >= chars:
                    break
        if not out:
            out.append('No matches for %r; showing the closest file instead.' % (pattern,))
        return '\n'.join(out) + '\n\n' + self.excerpt(rng, chars - used, hint=first.path if first else None)

    def glob(self, rng, pattern, chars):
        pattern = str(pattern or '*')
        matched = [s.path for s in self.sources if fnmatch.fnmatch(s.path, pattern)
                   or fnmatch.fnmatch(os.path.basename(s.path), pattern.split('/')[-1])]
        seen = set(matched)
        others = [s.path for s in self.sources if s.path not in seen]
        rng.shuffle(others)
        out, used = [], 0
        for path in matched + others:
            if used >= chars:
                break
            out.append('/home/dev/work/Tenstorrent.Blackhole-Qwen3.8-27B/' + path)
            used += len(out[-1]) + 1
        return '\n'.join(out)

    def tool_result(self, rng, name, arguments, chars):
        """What a tool returns to the agent, as real text of about `chars` characters."""
        args = arguments if isinstance(arguments, dict) else {}
        path = args.get('file_path') or args.get('path')
        if name == 'read':
            return self.excerpt(rng, chars, hint=path, offset=args.get('offset'))
        if name == 'grep':
            return self.grep(rng, args.get('pattern'), chars)
        if name == 'glob':
            return self.glob(rng, args.get('pattern'), chars)
        if name == 'bash':
            return '$ %s\n%s' % (str(args.get('command') or '')[:400], self.excerpt(rng, chars, hint=path))
        if name in ('edit', 'write'):
            return ('The file %s has been updated. Here is the result of running `cat -n` on a snippet of the '
                    'edited file:\n%s' % (path or 'the file', self.excerpt(rng, chars, hint=path)))
        if name == 'todowrite':
            return 'Todos have been modified successfully. Continue with the current task.'
        if name == 'task':
            return 'Sub-agent report (read-only search):\n' + self.excerpt(rng, chars)
        return self.excerpt(rng, chars, hint=path)

    def followup(self, rng, chars):
        text = FOLLOWUPS[rng.randrange(len(FOLLOWUPS))]
        return text + '\n```\n' + self.excerpt(rng, max(MIN_INPUT_CHARS, chars - len(text))) + '\n```'

    def compaction(self, rng, chars):
        return COMPACTION + self.excerpt(rng, chars) + '\n\nContinue with the last task where you left off.'


def parse_arguments(text):
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text or '{}')
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def complete_arguments(text):
    """True when a streamed tool call's argument string is a whole JSON object. vLLM parses the
    arguments of every assistant tool call it is sent back with json.loads (chat_utils.py:1855-1858),
    so one cut short would refuse the conversation's next turn."""
    if isinstance(text, dict):
        return True
    try:
        return isinstance(json.loads(text or '{}'), dict)
    except ValueError:
        return False


def assistant_message(result):
    """The assistant turn to send back, from the served answer: content (never None: the template
    trims it), reasoning (the field vLLM 0.25.1 returns and reads back), and the tool calls with the
    server's own ids and argument strings, in order. An answer cut off by max_tokens sends back no
    tool calls (its last call may be half-streamed), and a call whose arguments are not a whole JSON
    object is dropped: the next input is then a user follow-up instead of that call's result."""
    message = {'role': 'assistant', 'content': result.get('content') or ''}
    if result.get('reasoning'):
        message['reasoning'] = result['reasoning']
    calls = []
    for call in result.get('tool_calls') or ():
        if result.get('finish') == 'length':
            break
        function = call.get('function') or {}
        arguments = function.get('arguments') or '{}'
        if not complete_arguments(arguments):
            continue
        calls.append({'id': call.get('id') or 'call_%d' % len(calls), 'type': 'function',
                      'function': {'name': function.get('name') or '', 'arguments': arguments}})
    if calls:
        message['tool_calls'] = calls
    return message


class Conversation(object):
    """One agent conversation: its messages and tools, built turn by turn. The driver sends body(),
    appends the served answer with add_answer(), then calls extend() with the next input's token
    budget. fork() copies it for a variant (a changed suffix, an early divergence)."""

    def __init__(self, corpus, name, seed, system='full', first_tokens=0, task_messages=None):
        self.corpus, self.name, self.seed, self.system = corpus, name, str(seed), system
        self.tools = corpus.tools(system)
        self.messages = [{'role': 'system', 'content': corpus.system_text(system)}]
        self.turn = 0
        self.variant = ''
        if task_messages is not None:
            self.messages.extend(copy.deepcopy(task_messages))
        else:
            self.messages.append({'role': 'user', 'content': corpus.task(self.rng(), token_chars(first_tokens)
                                                                         if first_tokens else 0)})

    def rng(self, extra=''):
        return random.Random('%s|%s|%d|%s|%s' % (self.seed, self.name, self.turn, self.variant, extra))

    def body(self):
        return dict(messages=copy.deepcopy(self.messages), tools=copy.deepcopy(self.tools), tool_choice='auto')

    def last_assistant(self):
        for message in reversed(self.messages):
            if message['role'] == 'assistant':
                return message
        return None

    def add_answer(self, result):
        self.messages.append(assistant_message(result))
        self.turn += 1

    def pending_calls(self):
        """The tool calls of the last answer, if it is the last message."""
        if self.messages and self.messages[-1]['role'] == 'assistant':
            return self.messages[-1].get('tool_calls') or []
        return []

    def extend(self, input_tokens):
        """Append the next input at about `input_tokens` tokens: a tool result per tool call of the
        last answer (the budget shared out, todowrite small), or a user follow-up."""
        rng = self.rng('input')
        chars = token_chars(input_tokens)
        calls = self.pending_calls()
        if calls:
            weights = [0.05 if (call['function']['name'] == 'todowrite') else 1.0 for call in calls]
            total = sum(weights)
            for call, weight in zip(calls, weights):
                content = self.corpus.tool_result(rng, call['function']['name'],
                                                  parse_arguments(call['function']['arguments']),
                                                  max(MIN_INPUT_CHARS, int(chars * weight / total)))
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': content})
        else:
            self.messages.append({'role': 'user', 'content': self.corpus.followup(rng, chars)})

    def fork(self, variant):
        other = copy.copy(self)
        other.messages = copy.deepcopy(self.messages)
        other.tools = copy.deepcopy(self.tools)
        other.variant = variant
        return other

    def diverge_at(self, index, variant='early'):
        """A fork whose message `index` (a user or tool message) is replaced by other real text of
        about the same length: the prompt then diverges inside that message."""
        other = self.fork(variant)
        message = other.messages[index]
        if message['role'] not in ('user', 'tool'):
            raise ValueError('message %d is a %s message; only user and tool messages are replaced' % (
                index, message['role']))
        text = message['content']
        keep = min(len(text) // 4, 200)
        message['content'] = text[:keep] + self.corpus.excerpt(other.rng('diverge'), max(MIN_INPUT_CHARS, len(text) - keep))
        return other


def first_attachment(system, target_tokens):
    """The excerpt a first user message needs for its prompt to be about `target_tokens`."""
    return max(0, int(target_tokens) - SYSTEM_TOKENS[system])


def growth_input(target_tokens, prompt_tokens, completion_tokens, minimum=MIN_INPUT_CHARS / CHARS_PER_TOKEN):
    """The next input's token budget to reach `target_tokens` from a turn that was prompt + answer."""
    return max(int(minimum), int(target_tokens) - int(prompt_tokens) - int(completion_tokens))


def metering_input(rng, shape=METERING):
    """A tool result's size in the metering shape: lognormal around the mean, clipped."""
    value = rng.lognormvariate(0.0, 0.6) * shape['input_tokens_mean'] / 1.197   # E[lognormal(0, .6)] = 1.197
    return int(min(shape['input_tokens_max'], max(shape['input_tokens_min'], value)))


def think_gap(rng, mean_s):
    return rng.expovariate(1.0 / mean_s) if mean_s > 0 else 0.0


def chain_targets(first, hits):
    """A chained conversation's per-turn prompt targets: the first turn at `first`, then one turn
    at each of `hits` (sorted), so every hit turn is compared near a length the gate names."""
    return [int(first)] + sorted(int(hit) for hit in hits)


FILLERS = (' ok', '.', ' a', '\n')


def fit_tokens(build, count, target, max_chars, fillers=FILLERS, max_filler=48, back=24):
    """Build a prompt of exactly `target` tokens. build(pad_chars, filler_text) -> request body (the
    builder should strip the pad's trailing whitespace, where a filler would merge); count(body) ->
    its prompt token count (the server's /tokenize). A binary search finds the largest pad at or
    under the target, then repeated filler tokens close the gap; when every filler overshoots, the
    pad steps back a character and tries again. -> (body, tokens)."""
    lo, hi, best = 0, int(max_chars), None
    while lo <= hi:
        mid = (lo + hi) // 2
        tokens = count(build(mid, ''))
        if tokens <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise FitError('even an empty pad is %d tokens, past the %d asked' % (count(build(0, '')), target))
    for pad in range(best, max(-1, best - back - 1), -1):
        for filler in fillers:
            for fill in range(0, max_filler + 1):
                body = build(pad, filler * fill)
                tokens = count(body)
                if tokens == target:
                    return body, tokens
                if tokens > target:
                    break
    raise FitError('no pad and filler reached exactly %d tokens (best pad %d)' % (target, best))
