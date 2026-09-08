from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from lazy_ccl_links import rewrite


class LazyCclLinksTests(unittest.TestCase):
    def test_explicit_and_unspecified_native_selection(self):
        expression = 'num_links.value_or(ttnn::operations::ccl::common::get_num_links(device, axis))'
        rewritten = rewrite(expression, 1)
        self.assertEqual(rewritten, '(num_links.has_value() ? num_links.value() : ttnn::operations::ccl::common::get_num_links(device, axis))')
        with self.assertRaises(ValueError):
            rewrite(rewritten, 1)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_native_discovery_is_not_called_for_explicit_one_two_four(self):
        expression = rewrite('num_links.value_or(ttnn::operations::ccl::common::get_num_links(device, axis))', 1)
        source = '''#include <optional>
#include <cassert>
int calls = 0;
namespace ttnn::operations::ccl::common {
int get_num_links(int, int) { ++calls; return 1; }
}
int main() {
int device = 0, axis = 1;
for (int requested : {1, 2, 4}) {
std::optional<int> num_links = requested;
assert(EXPRESSION == requested);
assert(calls == 0);
}
std::optional<int> num_links;
assert(EXPRESSION == 1);
assert(calls == 1);
}
'''.replace('EXPRESSION', expression).replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'test.cpp').write_text(source)
            subprocess.run(['c++', '-std=c++17', str(path / 'test.cpp'), '-o', str(path / 'test')], check=True, capture_output=True)
            subprocess.run([str(path / 'test')], check=True, capture_output=True)
