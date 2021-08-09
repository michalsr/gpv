import argparse
import unittest

from exp.ours.experiments.cli_utils import MarkIfNotDefault


class TestCliUtil(unittest.TestCase):

  def test_mark_not_default(self):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, action=MarkIfNotDefault)

    args = parser.parse_args(["--batch_size", "32"])
    self.assertTrue(args.batch_size_not_default)
    self.assertEqual(args.batch_size, 32)

    args = parser.parse_args([])
    self.assertFalse(hasattr(args, "batch_size_not_default"))
    self.assertEqual(args.batch_size, 32)


if __name__ == '__main__':
  unittest.main()