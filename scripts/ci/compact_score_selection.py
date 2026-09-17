"""Host specification for exact finite-FP32 local winner reduction; no device route."""

import math
import struct


def ordered_key(bits):
    if type(bits) is not int or not 0 <= bits < 2 ** 32:
        raise ValueError('One uint32 FP32 encoding required')
    exponent = bits & 0x7f800000
    if exponent == 0x7f800000:
        raise ValueError('Nonfinite scores are outside this candidate specification')
    if bits & 0x7fffffff == 0:
        bits = 0
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000


def partitions(vocabulary, workers):
    if (type(vocabulary) is not int or type(workers) is not int
            or vocabulary < 1 or not 1 <= workers <= vocabulary):
        raise ValueError('Positive vocabulary and bounded worker count required')
    return [(worker * vocabulary // workers, (worker + 1) * vocabulary // workers)
            for worker in range(workers)]


def select(bits, workers):
    winners = []
    for start, end in partitions(len(bits), workers):
        winner = start
        maximum = ordered_key(bits[start])
        for token in range(start + 1, end):
            key = ordered_key(bits[token])
            if key > maximum:
                winner, maximum = token, key
        winners.append((maximum, winner))
    return max(winners, key=lambda item: (item[0], -item[1]))[1]


def fp32_bits(values):
    result = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError('Finite FP32 values required')
        result.append(struct.unpack('<I', struct.pack('<f', value))[0])
    return result
