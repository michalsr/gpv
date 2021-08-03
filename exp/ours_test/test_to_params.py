import json
import unittest
from typing import List, Optional, Dict, Union, Any, Tuple

from allennlp.common import Registrable, FromParams, Params
from dataclasses import dataclass

from exp.ours.util.to_params import to_params
import numpy as np


class RegisterableCls(Registrable):
  pass


@dataclass
class NonField(FromParams):
  f1: None = None


@dataclass
class AnyField(FromParams):
  f1: Any = None


@RegisterableCls.register("base-simple")
@dataclass
class BaseClassSimple(RegisterableCls):
  def __init__(self, f1: RegisterableCls):
    self.f1 = f1


@RegisterableCls.register("base-class1")
@dataclass
class BaseClass1(RegisterableCls):
  str_field: str = ""
  int_field: int = 0
  basic_lst: Optional[List[int]] = None
  basic_dict: Optional[Dict[str, int]] = None
  basic_dict_untyped: Optional[Dict] = None
  basic_list_untyped: Optional[List] = None


@RegisterableCls.register("nested")
@dataclass
class NestedClass(RegisterableCls):
  from_params_list: Optional[List[RegisterableCls]] = None
  from_params_dict: Optional[Dict[str, RegisterableCls]] = None
  from_params_field: Optional[RegisterableCls] = None


@RegisterableCls.register("base-class2")
@dataclass
class BaseClass2(RegisterableCls):
  def __init__(self, f1: str):
    self.f1 = f1


@dataclass
@RegisterableCls.register("union")
class BasicUnion(RegisterableCls):
  f1: Union[None, BaseClass1, BaseClass2, int, str] = None


@dataclass
@RegisterableCls.register("complex-union")
class ComplexUnion(RegisterableCls):
  f1: Union[int, Tuple[int, BaseClass1], BaseClass2] = None


class TestToParam(unittest.TestCase):

  def _test_to_params(self, x, cls=None):
    cls = x.__class__ if cls is None else cls
    params = to_params(x, cls)
    # Make sure it can be serialized
    json.dumps(Params(params).as_ordered_dict())
    new_instance = cls.from_params(Params(params))
    assert new_instance == x

  def _test_exception(self, x, cls=None):
    cls = x.__class__ if cls is None else cls
    self.assertRaises(ValueError, lambda: to_params(x, cls))

  def _rng_base1(self):
    return BaseClass1(int_field=np.random.randint(-100000, 100000))

  def _rng_base2(self):
    return BaseClass2(f1=str(np.random.randint(-100000, 100000)))

  def test_non_field(self):
    self._test_to_params(NonField())

  def test_any(self):
    self._test_to_params(AnyField("cat"))
    self._test_to_params(AnyField(0))
    self._test_exception(AnyField(object()))
    self._test_exception(AnyField([object()]))
    self._test_exception(AnyField(self._rng_base1()))

  def test_basic(self):
    self._test_to_params(BaseClass1(str_field="cat"), RegisterableCls)
    self._test_to_params(BaseClass1(int_field=0), BaseClass1)
    self._test_to_params(BaseClass1(basic_lst=[0, 1, 3]), RegisterableCls)
    self._test_to_params(BaseClass1(basic_dict=dict(a=1, b=2)), RegisterableCls)
    self._test_to_params(BaseClass1(basic_dict_untyped=dict(a=1, b=2)), RegisterableCls)
    self._test_to_params(BaseClass1(basic_list_untyped=["cat"]), RegisterableCls)

  def test_basic_error(self):
    self._test_exception(BaseClass1(str_field=object()))
    self._test_exception(BaseClass1(str_field=self._rng_base1()))
    self._test_exception(BaseClass1(basic_lst=[object()]))
    self._test_exception(BaseClass1(basic_dict_untyped=dict(a=[object()])))
    self._test_exception(BaseClass1(basic_lst=["cat", (1, self._rng_base1())]))

  def test_nested(self):
    base1 = NestedClass(from_params_list=[self._rng_base1(), self._rng_base1()])
    base2 = NestedClass(from_params_dict=dict(a=self._rng_base1(), b=self._rng_base1()))

    self._test_to_params(
      NestedClass(from_params_field=self._rng_base1()), RegisterableCls)
    self._test_to_params(
      NestedClass(from_params_field=self._rng_base2()), RegisterableCls)
    self._test_to_params(NestedClass(
      from_params_list=[self._rng_base1(), self._rng_base1()]), RegisterableCls)
    self._test_to_params(NestedClass(
      from_params_dict=dict(a=self._rng_base1(), b=self._rng_base1())), RegisterableCls)
    self._test_to_params(NestedClass(
      from_params_list=[base1, base1],
      from_params_dict=dict(a=base1, b=base2)
    ), RegisterableCls)

  def test_nested_error(self):
    self._test_exception(NestedClass(from_params_field=0))
    self._test_exception(NestedClass(from_params_list=[self._rng_base1(), 0]))
    self._test_exception(NestedClass(
      from_params_list=[self._rng_base1(), NestedClass(from_params_field=0)]
    ))

  def test_basic_union(self):
    self._test_to_params(BasicUnion(None))
    self._test_to_params(BasicUnion(0))
    self._test_to_params(BasicUnion("cat"))
    self._test_to_params(self._rng_base1())
    self._test_to_params(BaseClassSimple(BasicUnion("cat")), RegisterableCls)
    self._test_exception(BasicUnion(BasicUnion(0)))

  def test_complex_union(self):
    self._test_to_params(ComplexUnion(0))
    self._test_to_params(ComplexUnion(BaseClass2("dbg")))
    self._test_to_params(ComplexUnion((1, BaseClass1(int_field=0))))
    self._test_exception(BasicUnion(BasicUnion(0)))


if __name__ == '__main__':
  unittest.main()
