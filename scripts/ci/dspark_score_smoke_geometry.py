"""Isolated small simulator geometry; never production context admission."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import dspark_ladder_attention
import dspark_ladder_factory
import dspark_ladder_fixtures
import dspark_ladder_geometry
import dspark_ladder_score_center
import native_draft_sdpa


@contextmanager
def small_score_fixture():
    original_geometry = dspark_ladder_geometry.geometry
    original_center = dspark_ladder_score_center.scalar_score_center

    def geometry(context, output_tokens=1024):
        return original_geometry(context, 256 if context == 128 else output_tokens)

    @contextmanager
    def center(key_tiles=2112):
        with original_center(key_tiles=key_tiles):
            original = native_draft_sdpa.replacements

            def replacements():
                result = original()
                if key_tiles == 40:
                    result['compute_common.hpp'] = tuple((before, after.replace('== 40', '== 16'))
                        for before, after in result['compute_common.hpp'])
                return result

            with patch.object(native_draft_sdpa, 'replacements', replacements):
                yield

    with ExitStack() as stack:
        for module in (dspark_ladder_geometry, dspark_ladder_attention, dspark_ladder_fixtures):
            stack.enter_context(patch.object(module, 'geometry', geometry))
        stack.enter_context(patch.object(dspark_ladder_factory, 'KEY_TILES',
            tuple(sorted(set(dspark_ladder_factory.KEY_TILES) | {16}))))
        stack.enter_context(patch.object(dspark_ladder_score_center, 'scalar_score_center', center))
        yield
