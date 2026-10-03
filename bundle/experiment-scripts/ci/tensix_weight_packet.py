"""Opt-in fixed-packet compressed-weight reads; preserve every existing stream mapping and FIFO operation."""

import hashlib
from pathlib import Path

from attention_batch import Overlay


ORIGINAL_SHA256 = '733e58d5ef414069d3b67ce1d0058225e008aff9b72e8b37ed2f5cb087a2233b'


def packet_runtime(operations, engagements):
    original = Path(__file__).with_name('tensix_weight_stream_reader.cpp').resolve()
    candidate = Path(__file__).with_name('tensix_weight_packet_reader.cpp').resolve()
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_SHA256 or not isinstance(engagements, list):
        raise ValueError('Reviewed original reader and explicit engagement record required')

    def descriptor(**kwargs):
        if Path(kwargs['kernel_source']).resolve() == original:
            arguments = kwargs.get('compile_time_args', [])
            if (len(arguments) < 4 or type(arguments[0]) is not int or arguments[0] not in (576, 1088)
                    or arguments[1] != 8 or (arguments[2], arguments[3]) not in ((4, 272), (2, 160))):
                raise ValueError('Unchanged BF4/BF8 compressed-weight stream geometry required')
            kwargs = {**kwargs, 'kernel_source': str(candidate)}
            engagements.append(dict(tile_bytes=arguments[0], block_rows=arguments[1],
                receiver_columns=arguments[2], total_columns=arguments[3]))
        return operations.KernelDescriptor(**kwargs)

    return Overlay(operations, KernelDescriptor=descriptor)
