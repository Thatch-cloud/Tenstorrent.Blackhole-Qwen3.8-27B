"""Pinned sparse factory correction for exact rank-256 Markov partial reloads."""

import hashlib


SOURCE = 'ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp'
SOURCE_SHA256 = 'ba25c662b5b5486cf09e02d44e34ac7ffc742206aea2383a9e0d0cef720f9a46'
ANCHOR = '    // Create compute kernel\n'
INSERT = '''    const bool qwen_markov_fp32_reload =
        a.logical_shape() == ttnn::Shape{1, 1, 1, 256} &&
        (b.logical_shape() == ttnn::Shape{1, 1, 256, 64} ||
         b.logical_shape() == ttnn::Shape{1, 1, 256, 248320}) &&
        is_input_a_sparse && operation_attributes.is_input_b_sparse &&
        !nnz.has_value() && !use_indices && fp32_dest_acc_en && !packer_l1_acc_en &&
        in0_data_format == tt::DataFormat::Float16_b && in1_data_format == tt::DataFormat::Float16_b &&
        interm0_data_format == tt::DataFormat::Float32 && output_data_format == tt::DataFormat::Float32;
    std::vector<tt::tt_metal::UnpackToDestMode> qwen_unpack_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (qwen_markov_fp32_reload) {
        qwen_unpack_mode[tt::CBIndex::c_5] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

'''
CONFIG = '            .dst_full_sync_en = dst_full_sync_en,\n'
CONFIG_REPLACEMENT = CONFIG + '            .unpack_to_dest_mode = qwen_unpack_mode,\n'


def transform(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned original sparse factory required')
    if source.count(ANCHOR.encode()) != 1 or source.count(CONFIG.encode()) != 1:
        raise ValueError('Unique factory anchors required')
    return source.replace(ANCHOR.encode(), (INSERT + ANCHOR).encode()).replace(
        CONFIG.encode(), CONFIG_REPLACEMENT.encode())
