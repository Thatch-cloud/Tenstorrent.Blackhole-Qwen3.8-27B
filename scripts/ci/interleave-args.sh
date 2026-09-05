extra_args=(--limit-mm-per-prompt '{"image":0,"video":0}' --no-enable-mm-embeds)
if [ "$QWEN_INTERLEAVE_RATIO" != 0 ]; then
    export QWEN_PREFILL_CONTINUATION=1 TT_PREFILL_DECODE_INTERLEAVE=1
    export TT_DECODE_STEPS_PER_PREFILL_CHUNK="$QWEN_INTERLEAVE_RATIO"
    extra_args+=(--enable-chunked-prefill --max-num-batched-tokens 2048)
fi
