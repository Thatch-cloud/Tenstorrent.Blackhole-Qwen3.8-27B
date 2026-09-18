"""Explicit committed-block update for the pinned TT runner's single-request state."""


def apply_committed_output(runner, captured_state, output):
    request_id = output.request_id
    state = runner.requests.get(request_id)
    batch = runner.input_batch
    if state is not captured_state or state is None or batch.num_reqs != 1:
        raise ValueError('One live request with unchanged captured identity required')
    row = batch.req_id_to_index.get(request_id)
    if (row is None or row != 0 or batch.req_output_token_ids[row] is not state.output_token_ids
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
