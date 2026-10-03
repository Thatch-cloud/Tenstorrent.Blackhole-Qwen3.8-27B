import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("state", Path(__file__).with_name("summarize-gdn-state.py"))
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def fixture():
    records = []
    names = ("SliceDeviceOperation", "DecodeGatedDeltaRuleDeviceOperation",
             "InterleavedToShardedDeviceOperation", "SliceWriteDeviceOperation")
    for device in ("0", "1"):
        for index in range(48 * 4):
            offset = index % 4
            row = {"DEVICE ID": device, "METAL TRACE ID": "1", "METAL TRACE REPLAY SESSION ID": "2",
                   "GLOBAL CALL COUNT": str(index), "OP CODE": names[offset], "DEVICE KERNEL DURATION [ns]": "1000"}
            for tensor in ("INPUT_0", "INPUT_1", "INPUT_5", "OUTPUT_0", "OUTPUT_1"):
                full = (offset == 0 and tensor == "INPUT_0") or (offset == 3 and tensor in ("INPUT_1", "OUTPUT_0"))
                for axis, size in zip("WZYX", (8 if full else 1, 24, 128, 128)):
                    row[f"{tensor}_{axis}_PAD[LOGICAL]"] = f"{size}[{size}]"
                row[f"{tensor}_DATATYPE"] = "BFLOAT16"
            records.append(row)
    return records


class StateAttributionTests(unittest.TestCase):
    def run_summary(self, rows, native=None):
        if native is None:
            native = [dict(row, **{"OP NAME": row["OP CODE"]}) for row in rows]
        return state.summarize(rows, native, 1, {"2"})

    def test_complete_chain(self):
        result = self.run_summary(fixture())
        self.assertAlmostEqual(result["devices"][0]["median_copy_ms"], .144)
        self.assertAlmostEqual(result["devices"][0]["median_copy_fraction"], .75)

    def test_missing_native_call(self):
        rows = fixture()
        with self.assertRaisesRegex(AssertionError, "coverage"):
            self.run_summary(rows, [])

    def test_missing_chip(self):
        with self.assertRaisesRegex(AssertionError, "chip"):
            self.run_summary(fixture()[:192])

    def test_missing_layer(self):
        with self.assertRaisesRegex(AssertionError, "48"):
            self.run_summary(fixture()[4:])

    def test_wrong_adjacency(self):
        rows = fixture()
        rows[0]["OP CODE"] = "Other"
        with self.assertRaisesRegex(AssertionError, "adjacency"):
            self.run_summary(rows)

    def test_changed_precision(self):
        rows = fixture()
        rows[1]["INPUT_5_DATATYPE"] = "FLOAT32"
        with self.assertRaisesRegex(AssertionError, "precision"):
            self.run_summary(rows)

    def test_changed_pool(self):
        rows = fixture()
        rows[0]["INPUT_0_W_PAD[LOGICAL]"] = "1[1]"
        with self.assertRaisesRegex(AssertionError, "shape"):
            self.run_summary(rows)

    def test_invalid_time(self):
        rows = fixture()
        rows[0]["DEVICE KERNEL DURATION [ns]"] = "nan"
        with self.assertRaisesRegex(AssertionError, "duration"):
            self.run_summary(rows)


if __name__ == "__main__":
    unittest.main()
