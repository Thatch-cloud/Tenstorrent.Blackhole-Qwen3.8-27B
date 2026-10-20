// Syntax-check stub of windowed_mask_gen.hpp: the pinned writer's windowed-mask wrapper (inert here: use_windowed_mask is 0).
#pragma once
#include "stub_tt_api.hpp"

template <bool enabled, uint32_t cb_mask, uint32_t cb_cu_window, typename... Args>
void windowed_generate_if_enabled(Noc& noc, Args... args);
