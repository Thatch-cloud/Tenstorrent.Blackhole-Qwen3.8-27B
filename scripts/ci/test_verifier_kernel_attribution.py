import csv
import io
import json
import unittest

from verifier_kernel_attribution import selected_metadata


class KernelMetadataTests(unittest.TestCase):
    def stream(self, messages):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=['MessageName', 'total_ns'], delimiter=';', quotechar='`')
        writer.writeheader()
        for message in messages:
            writer.writerow(dict(MessageName=message, total_ns=1))
        stream.seek(0)
        return stream

    def template(self):
        data = dict(op_hash=77, device_id=0, global_call_count=10, op_code='GenericOpDeviceOperation',
            kernel_info=dict(compute_kernels=[dict(source='compute.cpp')], datamovement_kernels=[dict(source='reader.cpp')]),
            input_tensors=[dict(shape=dict(W='1', Z='1', Y='8', X='5120'))])
        return 'TT_DNN_OP ->\n' + json.dumps(data)

    def test_cached_selected_call_retains_only_bounded_metadata(self):
        source = self.stream([self.template(), 'TT_DNN_OP: Generic,77,0,true,11', 'TT_DNN_OP: Generic,77,0,true,12'])
        selected = selected_metadata(source, {(0, 11)})
        self.assertEqual(set(selected), {(0, 11)})
        self.assertEqual(selected[0, 11], ('GenericOpDeviceOperation', ('compute.cpp', 'reader.cpp'), (('1', '1', '8', '5120'),)))

    def test_missing_template_or_call_fails(self):
        for messages in ([], ['TT_DNN_OP: Generic,77,0,true,11'], [self.template()]):
            with self.assertRaises(ValueError):
                selected_metadata(self.stream(messages), {(0, 11)})

    def test_duplicate_conflicting_call_fails(self):
        other = self.template().replace('compute.cpp', 'different.cpp')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            selected_metadata(self.stream([self.template(), other]), {(0, 10)})
