#include "api/dataflow/dataflow_api.h"

// WP4 read-bandwidth probe: one data-movement kernel that reads a bfloat4_b or bfloat8_b interleaved weight the way the 1D matmul's weight reader does (one request per
// tile page) or in bank-contiguous multi-tile requests, with no compute and (in timing mode) no writes, so that the only thing that differs between two launches is the
// size and count of the DRAM requests.
//
// An interleaved DRAM tensor puts page p in bank p % banks at bank offset (p / banks) * page_bytes. With the tile columns a multiple of `banks`, the tiles (k, c),
// (k, c + banks), (k, c + 2 banks) ... of ONE tile row are consecutive pages of one bank, i.e. adjacent in that bank's memory: one NoC read of n * page_bytes from the
// address of page (k, c) returns n of them. A request descriptor is (c, n); n = 1 is the stock pattern. The worker's `requests` cover `run` tiles of every tile row, landing
// in L1 at row * run + (tiles before this request), so the landing zone of a block of `block_rows` rows is one contiguous region whatever the request sizes.
//
// Runtime args: [source address, destination address, request count, then (column, tiles) per request].
void kernel_main() {
    const uint32_t source_address = get_arg_val<uint32_t>(0);
    const uint32_t destination_address = get_arg_val<uint32_t>(1);
    const uint32_t requests = get_arg_val<uint32_t>(2);
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t row_pages = get_named_compile_time_arg_val("row_pages");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("page_bytes");
    constexpr uint32_t block_rows = get_named_compile_time_arg_val("block_rows");
    constexpr uint32_t run = get_named_compile_time_arg_val("run");
    constexpr uint32_t banks = get_named_compile_time_arg_val("banks");
    constexpr uint32_t write_back = get_named_compile_time_arg_val("write_back");
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, source_address, page_bytes);
    const auto destination = TensorAccessor(destination_args, destination_address, page_bytes);
    cb_reserve_back(0, 1);
    const uint32_t landing = get_write_ptr(0);
    for (uint32_t block = 0; block < rows / block_rows; ++block) {
        for (uint32_t row = 0; row < block_rows; ++row) {
            const uint32_t k = block * block_rows + row;
            uint32_t offset_tiles = 0;
            for (uint32_t request = 0; request < requests; ++request) {
                const uint32_t column = get_arg_val<uint32_t>(3 + 2 * request);
                const uint32_t tiles = get_arg_val<uint32_t>(4 + 2 * request);
                noc_async_read(source.get_noc_addr(k * row_pages + column), landing + (row * run + offset_tiles) * page_bytes, tiles * page_bytes);
                offset_tiles += tiles;
            }
        }
        noc_async_read_barrier();
        if (write_back) {
            for (uint32_t row = 0; row < block_rows; ++row) {
                const uint32_t k = block * block_rows + row;
                uint32_t offset_tiles = 0;
                for (uint32_t request = 0; request < requests; ++request) {
                    const uint32_t column = get_arg_val<uint32_t>(3 + 2 * request);
                    const uint32_t tiles = get_arg_val<uint32_t>(4 + 2 * request);
                    for (uint32_t tile = 0; tile < tiles; ++tile) {
                        noc_async_write_tile(k * row_pages + column + banks * tile, destination, landing + (row * run + offset_tiles + tile) * page_bytes);
                    }
                    offset_tiles += tiles;
                }
            }
            noc_async_write_barrier();
        }
    }
}
