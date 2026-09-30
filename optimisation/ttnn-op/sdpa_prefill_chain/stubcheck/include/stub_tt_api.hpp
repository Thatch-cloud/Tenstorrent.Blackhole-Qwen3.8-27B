// Syntax-check stubs for the tt-metal dataflow kernel API, as the served sdpa reader and
// dataflow_common.hpp (probe-v25) use it. NOT tt-metal: declarations only, shaped after the call
// sites, so that g++ -fsyntax-only can parse and type-check the served reader and the chain reader
// against the same surface. A pass is evidence that the chain reader uses the API the way the
// served reader does; the real JIT compile on card M (spec Q-12) is the proof.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#define FORCE_INLINE inline __attribute__((always_inline))
#define tt_l1_ptr
#define WAYPOINT(x) stub_waypoint(x)
#define ASSERT(...) stub_assert(__VA_ARGS__)

void stub_waypoint(const char (&name)[5]);
void stub_assert(bool condition);

constexpr uint32_t VALID = 1;
constexpr uint32_t INVALID = 0;

// Compile-time args: the served shape's reader CT vector (factory fd8c0676 with the chain flags
// suffix), injected by stub_compile.py as STUB_CT_ARGS.
constexpr uint32_t stub_ct_args[] = {STUB_CT_ARGS};
constexpr uint32_t get_compile_time_arg_val(uint32_t index) { return stub_ct_args[index]; }

template <typename T>
T get_arg_val(int index);
uint32_t get_semaphore(uint32_t semaphore_id);
void invalidate_l1_cache();
constexpr uint32_t get_tile_size(uint32_t cb_id) { return cb_id < 64 ? 1088u : 0u; }
extern uint8_t my_x[2];
extern uint8_t my_y[2];

enum class NocOptions : uint32_t { DEFAULT = 0, TXN_ID = 1 };

struct UnicastEndpoint {};
struct MulticastEndpoint {};

template <typename T>
class CoreLocalMem {
public:
    explicit CoreLocalMem(uint32_t address);
};

struct StubPageArgs {
    uint32_t page_id = 0;
    uint32_t offset_bytes = 0;
};
struct StubLocalArgs {
    uint32_t offset_bytes = 0;
};
struct StubUnicastArgs {
    uint32_t noc_x = 0;
    uint32_t noc_y = 0;
    uint32_t addr = 0;
};
struct StubMulticastArgs {
    uint32_t noc_x_start = 0;
    uint32_t noc_y_start = 0;
    uint32_t noc_x_end = 0;
    uint32_t noc_y_end = 0;
    uint32_t addr = 0;
};
struct StubTxnArgs {
    uint32_t trid = 0;
};

template <uint32_t Base>
struct TensorAccessorArgs {
    constexpr TensorAccessorArgs() {}
    constexpr uint32_t next_compile_time_args_offset() const { return Base + 1; }
};

template <typename Args>
class TensorAccessor {
public:
    TensorAccessor(const Args& args, uint32_t address, uint32_t page_size = 0);
    uint32_t get_aligned_page_size() const;
    uint32_t page_size = 0;
};

template <typename T, typename = void>
struct has_get_aligned_page_size : std::false_type {};
template <typename T>
struct has_get_aligned_page_size<T, std::void_t<decltype(std::declval<const T&>().get_aligned_page_size())>>
    : std::true_type {};
template <typename T>
constexpr bool has_get_aligned_page_size_v = has_get_aligned_page_size<T>::value;

class CircularBuffer {
public:
    explicit CircularBuffer(uint32_t cb_id);
    void reserve_back(uint32_t num_pages) const;
    void push_back(uint32_t num_pages) const;
    void wait_front(uint32_t num_pages) const;
    void pop_front(uint32_t num_pages) const;
    uint32_t get_write_ptr() const;
    uint32_t get_read_ptr() const;
    uint32_t get_cb_id() const;
};

class Noc {
public:
    Noc();
    uint8_t get_noc_id() const;
    template <typename Source>
    void async_read(const Source& src, CoreLocalMem<uint32_t> dst, uint32_t bytes, StubPageArgs src_args,
                    StubLocalArgs dst_args) const;
    void async_read(const UnicastEndpoint& src, CoreLocalMem<uint32_t> dst, uint32_t bytes, StubUnicastArgs src_args,
                    StubLocalArgs dst_args) const;
    template <NocOptions options = NocOptions::DEFAULT, typename Destination>
    void async_write(CoreLocalMem<uint32_t> src, const Destination& dst, uint32_t bytes, StubLocalArgs src_args,
                     StubPageArgs dst_args, StubTxnArgs txn = {}) const;
    template <NocOptions options = NocOptions::DEFAULT>
    void async_write(CoreLocalMem<uint32_t> src, const UnicastEndpoint& dst, uint32_t bytes, StubLocalArgs src_args,
                     StubUnicastArgs dst_args) const;
    template <NocOptions options = NocOptions::DEFAULT, typename Destination>
    void async_write(const CircularBuffer& src, const Destination& dst, uint32_t bytes, StubLocalArgs src_args,
                     StubPageArgs dst_args) const;
    void async_write_multicast(CoreLocalMem<uint32_t> src, const MulticastEndpoint& dst, uint32_t bytes,
                               uint32_t num_dests, StubLocalArgs src_args, StubMulticastArgs dst_args,
                               bool linked = false) const;
    void async_write_zeros(const CircularBuffer& cb, uint32_t bytes, StubLocalArgs dst_args) const;
    void async_read_barrier() const;
    template <NocOptions options = NocOptions::DEFAULT>
    void async_write_barrier(StubTxnArgs txn = {}) const;
    template <NocOptions options = NocOptions::DEFAULT>
    void async_writes_flushed(StubTxnArgs txn = {}) const;
    void async_atomic_barrier() const;
    void write_zeros_l1_barrier() const;
};

template <int CoreType = 0>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id);
    void set(uint32_t value) const;
    void wait(uint32_t value) const;
    void up(const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t increment) const;
    void relay_unicast(const Noc& noc, const Semaphore& remote, uint32_t noc_x, uint32_t noc_y) const;
    void relay_multicast(const Noc& noc, const Semaphore& remote, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                         uint32_t num_dests, bool linked) const;
    void set_multicast(const Noc& noc, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t num_dests) const;
};

namespace tt::constants {
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = 1024;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HW = 256;
}  // namespace tt::constants
