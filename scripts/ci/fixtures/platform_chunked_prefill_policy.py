import os


_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}

def _apply_chunked_prefill_policy(vllm_config: "VllmConfig") -> None:
    """Restrict token-chunked prefill to the model types that support it."""
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config
    model_type = getattr(model_config.hf_config, "model_type", None)

    if model_type in _CHUNKED_PREFILL_MODEL_TYPES:
        # A chunk boundary inside a multimodal item would split its embeddings
        # from their positions. Only meaningful while prefill can be split, and
        # vLLM rejects the flag outright when one item exceeds the token budget,
        # so it stays off for every model type below.
        scheduler_config.disable_chunked_mm_input = True
        return

    if scheduler_config.enable_chunked_prefill:
        logger.info(
            "Chunked prefill is not supported for `model_type=%s`; disabling it.",
            model_type,
        )
        scheduler_config.enable_chunked_prefill = False

        # vLLM does this bump silently earlier if chunked prefill is already
        # disabled and max_num_batched_tokens is not explicitly set. We can't
        # know if it was specified or the default, hence the warning.
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_model_len = model_config.max_model_len
        if max_num_batched_tokens < max_model_len:
            logger.warning(
                "max_num_batched_tokens=%d < max_model_len=%d with chunked "
                "prefill disabled, bumping max_num_batched_tokens to match.",
                max_num_batched_tokens,
                max_model_len,
            )
            scheduler_config.max_num_batched_tokens = max_model_len

    # The base scheduler caps a prefill at this threshold before it consults
    # ``enable_chunked_prefill``, so leaving it nonzero still splits prefills
    # for a model that cannot resume one.
    scheduler_config.long_prefill_token_threshold = 0
