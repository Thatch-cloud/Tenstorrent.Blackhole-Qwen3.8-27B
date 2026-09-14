"""Unqualified, explicitly selected FP32 split-K intermediate factory transform."""

SOURCE = 'ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp'


def transform(source):
    precision = '''    const char* qwen_splitk_precision = std::getenv("QWEN_SPLITK_FP32_INTERMEDIATES");
    const bool qwen_splitk_fp32 = qwen_splitk_precision && std::string(qwen_splitk_precision) == "1";
    TT_FATAL(!qwen_splitk_fp32 || (B == 4 && PNH == 128 && num_kv_heads == 1 &&
        DH == 128 && vDH == 128 && fp32_dest_acc_en && !is_causal && use_attention_mask),
        "Explicit folded split-K geometry required for FP32 intermediates");
    const tt::DataFormat im_df = qwen_splitk_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const tt::DataFormat stats_df = qwen_splitk_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;'''
    modes = '''    std::vector<tt::tt_metal::UnpackToDestMode> qwen_splitk_unpack_modes;
    if (qwen_splitk_fp32) {
        qwen_splitk_unpack_modes.resize(32, tt::tt_metal::UnpackToDestMode::Default);
        for (uint32_t cb : {4u, 6u, 7u, 16u, 17u, 18u, 19u, 21u, 22u, 23u,
                24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u}) {
            qwen_splitk_unpack_modes.at(cb) = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }
    compute_desc.config = ComputeConfigDescriptor{'''
    substitutions = (
        ('    const tt::DataFormat im_df = tt::DataFormat::Float16_b;\n'
         '    const tt::DataFormat stats_df = tt::DataFormat::Float16_b;', precision),
        ('    compute_desc.config = ComputeConfigDescriptor{', modes),
        ('        .dst_full_sync_en = dst_full_sync_en,\n        .math_approx_mode = math_approx_mode,',
         '        .dst_full_sync_en = dst_full_sync_en,\n'
         '        .unpack_to_dest_mode = qwen_splitk_unpack_modes,\n        .math_approx_mode = math_approx_mode,'))
    for before, after in substitutions:
        if source.count(before) != 1:
            raise ValueError('Exact decode factory anchor required')
        source = source.replace(before, after)
    return source
