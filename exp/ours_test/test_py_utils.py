import unittest

import torch

from exp.ours.util import py_utils
from exp.ours.util.py_utils import ReplaceAll


class TestPyUtils(unittest.TestCase):

  def test_balanced_merge_same_lens(self):
    out = py_utils.balanced_merge_multi([
      list(range(0, 3)),
      list(range(3, 6)),
      list(range(6, 9))
    ])
    self.assertEqual(set(out[:3]), {0, 3, 6})
    self.assertEqual(set(out[3:6]), {1, 4, 7})
    self.assertEqual(set(out[6:9]), {2, 5, 8})

  def test_balanced_merge_different_lens(self):
    out = py_utils.balanced_merge_multi([
      list(range(0, 6)),
      list(range(30, 33)),
      list(range(100, 118))
    ])
    self.assertEqual(set(out[:9]), {0, 1, 30} | set(range(100, 106)))
    self.assertEqual(set(out[9:18]), {2, 3, 31} | set(range(106, 112)))
    self.assertEqual(set(out[18:]), {4, 5, 32} | set(range(112, 118)))

  def test_replace_all(self):
    rall = ReplaceAll({"cat": "c", "dog": "d", "cats": "cs"})
    self.assertEqual(rall.replace(""), "")
    self.assertEqual(rall.replace("cat"), "c")
    self.assertEqual(rall.replace("catc"), "catc")
    self.assertEqual(rall.replace("cat cat"), "c c")
    self.assertEqual(rall.replace("cat cats"), "c cs")
    self.assertEqual(rall.replace("dog cats"), "d cs")
    self.assertEqual(rall.replace("a cats"), "a cs")
    self.assertEqual(rall.replace("cats a"), "cs a")
    self.assertEqual(rall.replace("catsdog"), "catsdog")
    self.assertEqual(rall.replace("cats,dog"), "cs,d")

  def test_replace_all_rescape(self):
    rall = ReplaceAll({"[x]": "x"})
    self.assertEqual(rall.replace("[x]"), "x")
    self.assertEqual(rall.replace("[x] cat"), "x cat")
    self.assertEqual(rall.replace("cat [x]"), "cat x")

  def test_flat_to_nested(self):
    r = lambda: torch.empty(1).uniform_(-1, 1)

    for test_case in [
      [r()],
      [r(), r()],
      {"a": r()},
      {"a": [r()], "b": r()},
      [r(), r(), [r()]],
      [{"a": [r(), r(), r()], "b": r()}, r(), [r()]],
      {"a": {"c": r(), "a": [r()]}, "b": r()},
    ]:
      flat = py_utils.nested_struct_to_flat(test_case)
      rebuilt = py_utils.flat_to_nested_struct(flat)

      # `self.assertEqual` does not seem to work correctly for tensors, so manually check with `==`
      self.assertTrue(rebuilt == test_case)


if __name__ == '__main__':
  unittest.main()
  # TestPyUtils().run()