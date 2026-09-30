"""Lever N M1, step 1: what one resumable prefill step must do.

Pure planning logic, no device calls, so the contract can be pinned down and tested
before any of it runs on hardware. model.prefill_paged_slots_range consumes a plan
per request and performs the device work; everything decided here is decidable on the
host.

The contract is docs/lever-N-prefill-decode-interleave.md section 3.1:

    for each row u: process prompt tokens [start, end) of request u
      start == 0        -> first chunk: reset GDN state + stage the request's RoPE
      end == prompt_len -> last chunk: tail + logits + slot write
      otherwise         -> intermediate: no logits (return zeros), no slot write

Invariants the design leans on, asserted here rather than papered over:

- Continuation starts are whole chunks. vLLM's per-step budget must equal the model's
  chunk size, so max_num_batched_tokens == chunk_size and start % chunk_size == 0. A
  mismatch is a configuration error.
- The tail (a partial chunk) is only ever processed on the last step, because it is
  what produces the logits and the slot write.
- A preempted partial prefill resumes at start == 0. The plugin replays the prompt
  plus generated tokens, so a fresh first chunk is the correct handling, not an error.
"""

DEFAULT_CHUNK_SIZE = 2048


class PrefillStepPlan:
    """What one request needs from one prefill step."""

    __slots__ = ('start', 'end', 'prompt_len', 'chunk_size', 'first', 'last',
                 'full_chunk_indices', 'tail_start', 'tail_tokens')

    def __init__(self, start, end, prompt_len, chunk_size):
        self.start, self.end = start, end
        self.prompt_len, self.chunk_size = prompt_len, chunk_size
        self.first = start == 0
        self.last = end == prompt_len
        num_full_total = prompt_len // chunk_size
        first_chunk = start // chunk_size
        # Whole chunks wholly inside this step's window.
        last_chunk = min(end // chunk_size, num_full_total)
        self.full_chunk_indices = tuple(range(first_chunk, last_chunk))
        self.tail_start = num_full_total * chunk_size
        self.tail_tokens = (prompt_len - self.tail_start) if self.last else 0

    @property
    def resets_state(self):
        """Only the first step resets GDN state and stages the request's RoPE."""
        return self.first

    @property
    def emits_logits(self):
        """Intermediate steps return zeros; only the last step produces a next-token logit."""
        return self.last

    @property
    def writes_slot(self):
        """The decode slot is written once, from the last step's snapshot."""
        return self.last

    @property
    def runs_tail(self):
        return self.last and self.tail_tokens > 0

    def describe(self):
        return dict(start=self.start, end=self.end, prompt_len=self.prompt_len,
                    first=self.first, last=self.last,
                    full_chunks=list(self.full_chunk_indices),
                    tail_start=self.tail_start, tail_tokens=self.tail_tokens,
                    resets_state=self.resets_state, emits_logits=self.emits_logits,
                    writes_slot=self.writes_slot)


def plan_step(start, end, prompt_len, chunk_size=DEFAULT_CHUNK_SIZE):
    """Plan one request's slice of one prefill step.

    start/end are the half-open token window this step covers, in the runner's terms:
    end is the chunk END, not the prompt length.
    """
    for name, value in (('start', start), ('end', end), ('prompt_len', prompt_len),
                        ('chunk_size', chunk_size)):
        if type(value) is not int:
            raise ValueError('%s must be an int' % name)
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if prompt_len < 1:
        raise ValueError('prompt_len must be at least one token')
    if not 0 <= start < end <= prompt_len:
        raise ValueError('require 0 <= start < end <= prompt_len, got %d, %d, %d'
                         % (start, end, prompt_len))
    if start % chunk_size:
        raise ValueError('continuation must start on a chunk boundary: %d %% %d != 0'
                         % (start, chunk_size))
    if end != prompt_len and end % chunk_size:
        raise ValueError('a non-final step must end on a chunk boundary: %d %% %d != 0'
                         % (end, chunk_size))
    return PrefillStepPlan(start, end, prompt_len, chunk_size)


def plan_batch(starts, ends, prompt_lens, chunk_size=DEFAULT_CHUNK_SIZE):
    lengths = {len(starts), len(ends), len(prompt_lens)}
    if len(lengths) != 1:
        raise ValueError('starts, ends and prompt_lens must be the same length')
    return [plan_step(int(s), int(e), int(p), chunk_size)
            for s, e, p in zip(starts, ends, prompt_lens)]


def validate_chunking(max_num_batched_tokens, chunk_size=DEFAULT_CHUNK_SIZE):
    """The scheduler budget must equal the model's chunk size.

    A larger budget hands the model a window it cannot replay as whole traced chunks;
    a smaller one starts a continuation mid-chunk. Both are configuration errors.
    """
    if type(max_num_batched_tokens) is not int or max_num_batched_tokens != chunk_size:
        raise ValueError('max_num_batched_tokens must equal the model chunk size %d, got %r'
                         % (chunk_size, max_num_batched_tokens))
    return True


def steps_for_prompt(prompt_len, chunk_size=DEFAULT_CHUNK_SIZE):
    """The window sequence a whole prompt is served in, for tests and for reasoning."""
    if type(prompt_len) is not int or prompt_len < 1:
        raise ValueError('prompt_len must be a positive int')
    windows, start = [], 0
    while start < prompt_len:
        end = min(start + chunk_size, prompt_len)
        windows.append((start, end))
        start = end
    return windows
