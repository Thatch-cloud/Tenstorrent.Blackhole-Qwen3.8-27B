import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("shapes", Path(__file__).with_name("summarize-model-shapes.py"))
shapes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shapes)


class ShapeTests(unittest.TestCase):
    def test_native_coverage_and_shape_ranking(self):
        rows = [dict(zip(("METAL TRACE ID", "METAL TRACE REPLAY SESSION ID", "DEVICE ID", "OP CODE",
                          "INPUT_1_Y_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]", "CORE COUNT", "INPUT_1_DATATYPE",
                          "DEVICE KERNEL DURATION [ns]"),
                         ("1", replay, "0", "MatmulDeviceOperation", "5120[5120]", "1024[1024]", "32", "BFLOAT8_B", "1000")))
                for replay in ("2", "3")]
        native = [dict(row, **{"OP NAME": row["OP CODE"]}) for row in rows]
        result = shapes.summarize(rows, native, 1, {"2", "3"})
        self.assertEqual(result[0]["median_summed_kernel_ms"], .001)
        with self.assertRaises(AssertionError):
            shapes.summarize(rows[:-1], native, 1, {"2", "3"})


if __name__ == "__main__":
    unittest.main()
