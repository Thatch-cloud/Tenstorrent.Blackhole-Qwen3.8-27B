"""Explicit text-fixture-only construction scope; never a serving default."""

from contextlib import contextmanager
import os
from unittest.mock import patch


@contextmanager
def text_only_load(model_class, records):
    if os.environ.get('QWEN_TEXT_ONLY_LOAD') != '1':
        raise ValueError('Explicit text-only loading experiment required')
    original = model_class.init_vision_model
    models = []
    record = dict(scope=__doc__, skipped_vision_initializations=0, restored=False,
        performance_qualified=False, correctness_qualified=False, serving_qualified=False)
    records.append(record)

    def skip(model, *args, **kwargs):
        if args or kwargs or getattr(model, 'vision_model', 'missing') is not None:
            raise ValueError('Only fresh default vision initialization may be skipped')
        if models:
            raise ValueError('Only one text model may be constructed in this experiment')
        models.append(model)
        record['skipped_vision_initializations'] += 1
        return None

    with patch.object(model_class, 'init_vision_model', skip):
        try:
            yield
            if len(models) != 1 or models[0].vision_model is not None:
                raise ValueError('Exactly one vision-free text model required')
        finally:
            if model_class.init_vision_model is not skip:
                raise RuntimeError('Text-only initialization hook changed externally')
    record['restored'] = model_class.init_vision_model is original
