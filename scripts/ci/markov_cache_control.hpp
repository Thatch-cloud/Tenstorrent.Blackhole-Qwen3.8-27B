#pragma once

#include <cstdint>
#include <limits>

namespace markov_cache {

constexpr uint32_t slots = 64;
constexpr uint32_t vocabulary = 248320;
constexpr uint32_t invalid_slot = std::numeric_limits<uint32_t>::max();

struct Slot {
    uint32_t valid;
    uint32_t token;
    uint32_t epoch;
    uint32_t ticket;
    uint32_t reserved[4];
};

struct State {
    uint32_t epoch;
    uint32_t clock;
    uint32_t reserved[6];
    Slot entries[slots];
};

struct Decision {
    uint32_t ok;
    uint32_t hit;
    uint32_t slot;
    uint32_t token;
    uint32_t epoch;
    uint32_t ticket;
    uint32_t reserved[2];
};

static_assert(sizeof(Slot) == 32);
static_assert(sizeof(Decision) == 32);
static_assert(sizeof(State) == (slots + 1) * 32);

inline Decision lookup(State& state, uint32_t token, uint32_t epoch) {
    Decision result{0, 0, invalid_slot, token, epoch, 0, {0, 0}};
    if (token >= vocabulary || epoch == 0 || epoch < state.epoch) {
        return result;
    }
    if (state.epoch != epoch) {
        state.epoch = epoch;
        state.clock = 0;
        for (auto& entry : state.entries) {
            entry.valid = 0;
        }
    }
    if (state.clock == std::numeric_limits<uint32_t>::max()) {
        return result;
    }
    result.ticket = ++state.clock;
    uint32_t victim = invalid_slot;
    uint32_t oldest = std::numeric_limits<uint32_t>::max();
    for (uint32_t index = 0; index < slots; ++index) {
        auto& entry = state.entries[index];
        if (entry.valid == 1 && entry.epoch == epoch && entry.token == token) {
            entry.ticket = result.ticket;
            result.ok = result.hit = 1;
            result.slot = index;
            return result;
        }
        if (entry.valid != 1 || entry.epoch != epoch) {
            if (oldest != 0) {
                victim = index;
                oldest = 0;
            }
        } else if (entry.ticket < oldest) {
            oldest = entry.ticket;
            victim = index;
        }
    }
    if (victim == invalid_slot) {
        return result;
    }
    state.entries[victim] = Slot{0, token, epoch, result.ticket, {0, 0, 0, 0}};
    result.ok = 1;
    result.slot = victim;
    return result;
}

inline bool commit(State& state, const Decision& decision) {
    if (decision.ok != 1 || decision.hit != 0 || decision.slot >= slots ||
        decision.epoch == 0 || decision.epoch != state.epoch || decision.ticket != state.clock) {
        return false;
    }
    auto& entry = state.entries[decision.slot];
    if (entry.valid != 0 || entry.token != decision.token || entry.epoch != decision.epoch ||
        entry.ticket != decision.ticket) {
        return false;
    }
    entry.valid = 1;
    return true;
}

}
