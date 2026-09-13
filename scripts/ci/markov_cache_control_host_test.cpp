#include "markov_cache_control.hpp"

#include <cassert>

int main() {
    using namespace markov_cache;
    State state{};
    auto first = lookup(state, 198, 1);
    assert(first.ok && !first.hit);
    auto uncommitted = lookup(state, 198, 1);
    assert(!uncommitted.hit);
    assert(!commit(state, first));
    assert(commit(state, uncommitted));
    assert(!commit(state, uncommitted));
    auto hit = lookup(state, 198, 1);
    assert(hit.ok && hit.hit);
    assert(!commit(state, hit));

    for (uint32_t token = 0; token < slots; ++token) {
        auto decision = lookup(state, token, 2);
        assert(decision.ok && !decision.hit);
        assert(commit(state, decision));
    }
    assert(lookup(state, 0, 2).hit);
    auto replacement = lookup(state, 100, 2);
    assert(replacement.slot == 1);
    assert(commit(state, replacement));
    assert(!lookup(state, 1, 2).hit);

    auto old_epoch = lookup(state, 123, 2);
    auto changed_weights = lookup(state, 123, 3);
    assert(!changed_weights.hit);
    assert(!commit(state, old_epoch));
    assert(commit(state, changed_weights));
    assert(!lookup(state, 123, 2).ok);
    assert(!commit(state, changed_weights));

    assert(!lookup(state, vocabulary, 2).ok);
    assert(!lookup(state, 0, 0).ok);
    state.clock = std::numeric_limits<uint32_t>::max();
    assert(!lookup(state, 0, 2).ok);
    assert(lookup(state, 0, 4).ok);
    return 0;
}
