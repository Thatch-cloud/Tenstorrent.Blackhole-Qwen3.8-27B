"""Explicit committed-block update for the pinned TT runner's single-request state."""


def validate_runner_reservation(runner, captured_state, ticket, users=1):
    # `users` is how many requests share this scheduled step. At one it is exactly
    # the old contract; packed, the batch holds one row per user and a request's
    # row is whichever index the batch gave it, not row zero.
    state = runner.requests.get(ticket.request_id)
    batch = runner.input_batch
    row = batch.req_id_to_index.get(ticket.request_id)
    if (state is not captured_state or state is None or batch.num_reqs != users
            or row is None or not 0 <= row < users
            or batch.req_output_token_ids[row] is not state.output_token_ids
            or state.prompt_token_ids is None):
        raise ValueError('One live captured request and stable output binding required')
    start = int(batch.num_tokens[row])
    if (start != ticket.position + 1 or start != len(state.prompt_token_ids) + len(state.output_token_ids)
            or int(batch.num_computed_tokens_cpu[row]) != ticket.position
            or state.num_computed_tokens != ticket.position):
        raise ValueError('Runner and prepared target frontier differ before verification')
    end = start + len(ticket.tokens)
    if end > runner.model_config.max_model_len or end > batch.token_ids_cpu.shape[1]:
        raise ValueError('Runner storage cannot hold the maximum committed block')


def apply_committed_output(runner, captured_state, output, users=1):
    request_id = output.request_id
    state = runner.requests.get(request_id)
    batch = runner.input_batch
    if state is not captured_state or state is None or batch.num_reqs != users:
        raise ValueError('One live request with unchanged captured identity required')
    row = batch.req_id_to_index.get(request_id)
    if (row is None or not 0 <= row < users
            or batch.req_output_token_ids[row] is not state.output_token_ids
            or state.prompt_token_ids is None):
        raise ValueError('Stable single-request token row and shared output-list binding required')
    tokens = tuple(output.token_ids)
    if (len(tokens) > 16 or any(type(token) is not int or not 0 <= token < runner.model_config.get_vocab_size()
            for token in tokens) or type(output.position) is not int
            or (output.cancelled and tokens) or (not tokens and not output.finished)):
        raise ValueError('Only bounded committed global token IDs or terminal empty output allowed')
    start = int(batch.num_tokens[row])
    if (start != len(state.prompt_token_ids) + len(state.output_token_ids)
            or int(batch.num_computed_tokens_cpu[row]) != start - 1
            or state.num_computed_tokens != start - 1
            or output.position != start + len(tokens) - 1):
        raise ValueError('Committed output must advance the exact pre-step target frontier once')
    end = start + len(tokens)
    if end > runner.model_config.max_model_len or end > batch.token_ids_cpu.shape[1]:
        raise ValueError('Committed block exceeds reserved runner token storage')
    if not tokens:
        return
    batch.token_ids_cpu[row, start:end] = tokens
    state.output_token_ids.extend(tokens)
    batch.num_tokens[row] = end
    batch.num_computed_tokens_cpu[row] = output.position
    state.num_computed_tokens = output.position
