#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t address = get_arg_val<uint32_t>(0);
    const uint32_t worker = get_arg_val<uint32_t>(1);
    const uint32_t first_x = get_arg_val<uint32_t>(2);
    const uint32_t first_y = get_arg_val<uint32_t>(3);
    const uint32_t last_x = get_arg_val<uint32_t>(4);
    const uint32_t last_y = get_arg_val<uint32_t>(5);
    const uint32_t workers = get_arg_val<uint32_t>(6);
    const uint32_t receivers = get_arg_val<uint32_t>(7) - 1;
    constexpr auto tensor_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(tensor_args, address, 2048);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    for (uint32_t block = 0; block < 20; ++block) {
        cb_reserve_back(0, 8);
        const uint32_t destination = get_write_ptr(0);
        if (worker == 0) {
            for (uint32_t tile = 0; tile < 8; ++tile) {
                noc_async_read_tile(block * 8 + tile, input, destination + tile * 2048);
            }
            noc_async_read_barrier();
            noc_semaphore_wait(ready, receivers);
            noc_semaphore_set(ready, 0);
            const uint64_t target = get_noc_multicast_addr(last_x, last_y, first_x, first_y, destination);
            noc_async_write_multicast(destination, target, 8 * 2048, receivers);
            noc_async_write_barrier();
            noc_semaphore_set(received, 1);
            const uint64_t signal = get_noc_multicast_addr(last_x, last_y, first_x, first_y, get_semaphore(1));
            noc_semaphore_set_multicast(get_semaphore(1), signal, receivers);
        } else {
            noc_semaphore_inc(get_noc_addr(first_x, first_y, get_semaphore(0)), 1);
            noc_semaphore_wait(received, 1);
            noc_semaphore_set(received, 0);
        }
        cb_push_back(0, 8);
        if (worker >= workers) {
            cb_wait_front(0, 8);
            cb_pop_front(0, 8);
        }
    }
}
