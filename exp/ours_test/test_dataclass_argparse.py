import random
import unittest
from typing import List, Optional, Union

from dataclasses import dataclass, replace

from exp.ours.old.dataclass_argparse import update_with_args


@dataclass
class Basic:
  i: int
  s: str
  i2: Optional[int]
  f: float
  il: List[int]
  d: str = "default"


@dataclass
class IHolder:
  i: int

@dataclass
class SHolder:
  s: str


@dataclass
class Nested:
  a: IHolder
  b: str
  c: IHolder


@dataclass
class Choice:
  choice: Union[IHolder, SHolder]


class TestDataclassArgparse(unittest.TestCase):

  def test_basic(self):
    basic = Basic(0, "", None, 0, [0])
    update_with_args(basic, ["--i", "2", "--s", "cat"])
    expected = replace(basic, i=2, s="cat")
    self.assertEqual(basic, expected)

    update_with_args(basic, ["--il", "2", "4"])
    self.assertEqual(basic, replace(expected, il=[2, 4]))

  def _rng_basic(self):
    return IHolder(random.randint(0, 1000))

  def test_nested(self):
    initial = Nested(self._rng_basic(), "dbg", self._rng_basic())
    update_with_args(initial, ["--a.i", "2"])
    self.assertEqual(initial.a.i, 2)

    update_with_args(initial, ["--a.i", "-2", "--c.i", "4", "--b", "test"])
    self.assertEqual(initial, Nested(IHolder(-2), "test", IHolder(4)))

  def test_choice_small(self):
    # noinspection PyTypeChecker
    c = Choice(dict(ih=IHolder(2), sh=SHolder("")))
    update_with_args(c, ["--choice", "--ih"])
    print(c)


if __name__ == '__main__':
  TestDataclassArgparse().test_choice_small()