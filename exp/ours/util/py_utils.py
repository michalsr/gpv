import logging
import sys
import zlib
from multiprocessing import Lock, Pool

import torch
from collections import defaultdict
from os import listdir, remove
from os.path import exists, isdir, join
from shutil import rmtree
from typing import TypeVar, List, Iterable, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm


class TerminalColors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'


def get_yes_no(msg):
  while True:
    txt = input(msg).strip().lower()
    if txt in {"y", "yes"}:
      return True
    if txt in {"n", "no"}:
      return False


def clear_if_nonempty(output_dir, override=False):
  if output_dir:
    if exists(output_dir) and listdir(output_dir):
      if override or get_yes_no("%s is non-empty, override (y/n)?" % output_dir):
        for x in listdir(output_dir):
          if isdir(join(output_dir, x)):
            rmtree(join(output_dir, x))
          else:
            remove(join(output_dir, x))
      else:
        raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")


def add_stdout_logger():
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%m/%d %H:%M:%S', )
  handler.setFormatter(formatter)
  handler.setLevel(logging.DEBUG)

  root = logging.getLogger()
  root.setLevel(logging.INFO)
  root.addHandler(handler)

  # Re-direction warning to  logging
  logging.captureWarnings(True)


class DisableLogging:

  def __init__(self, to_level=logging.INFO):
    self.to_level = to_level

  def __enter__(self):
    self.prev_level = logging.root.manager.disable
    if self.prev_level < self.to_level:
      logging.disable(self.to_level)

  def __exit__(self, exc_type, exc_val, exc_tb):
    logging.disable(self.prev_level)


def duration_to_str(seconds):
  sign_string = '-' if seconds < 0 else ''
  seconds = abs(int(seconds))
  days, seconds = divmod(seconds, 86400)
  hours, seconds = divmod(seconds, 3600)
  minutes, seconds = divmod(seconds, 60)
  if days > 0:
    return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
  elif hours > 0:
    return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
  elif minutes > 0:
    return '%s%dm%ds' % (sign_string, minutes, seconds)
  else:
    return '%s%ds' % (sign_string, seconds)


T = TypeVar('T')
K = TypeVar('K')


def transpose_lists(lsts: Iterable[Iterable[T]]) -> List[List[T]]:
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def transpose_list_of_dicts(lst: List[Dict[K, T]]) -> Dict[K, List[T]]:
  out = defaultdict(list)
  for r in lst:
    for k, v in r.items():
      out[k].append(v)
  return {k: v for k, v in out.items()}


def transpose_dicts(dicts: Dict[K, Dict[T, Any]]) -> Dict[T, Dict[K, Any]]:
  out = defaultdict(dict)
  for k1, v1 in dicts.items():
    for k2, v2 in v1.items():
      out[k2][k1] = v2
  return out


def consistent_hash(x):
  return zlib.crc32(x)


def split_list(lst: List[T], n_groups) -> List[List[T]]:
  """ partition `lst` into `n_groups` that are as evenly sized as possible  """
  per_group = len(lst) // n_groups
  remainder = len(lst) % n_groups
  groups = []
  ix = 0
  for _ in range(n_groups):
    group_size = per_group
    if remainder > 0:
      remainder -= 1
      group_size += 1
    groups.append(lst[ix:ix + group_size])
    ix += group_size
  return groups


def group_list(lst: List[T], max_group_size) -> List[List[T]]:
  """partition `lst` into that the mininal number of groups that as evenly sized
  as possible  and are at most `max_group_size` in size """
  if max_group_size is None:
    return [lst]
  n_groups = (len(lst) + max_group_size - 1) // max_group_size
  per_group = len(lst) // n_groups
  remainder = len(lst) % n_groups
  groups = []
  ix = 0
  for _ in range(n_groups):
    group_size = per_group
    if remainder > 0:
      remainder -= 1
      group_size += 1
    groups.append(lst[ix:ix + group_size])
    ix += group_size
  return groups


def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
  """Unpack lists into a single list."""
  return [x for sublist in iterable_of_lists for x in sublist]


def partition_list(lst: List[T], fn) -> Tuple[List[T], List[T]]:
  is_true = []
  is_false = []
  for x in lst:
    if fn(x):
      is_true.append(x)
    else:
      is_false.append(x)
  return is_true, is_false


def sample_dict(x: Dict[K, T], sample: Optional[int]) -> Dict[K, T]:
  if sample is None or len(x) < sample:
    return x
  keys = list(x)
  to_keep = np.random.choice(keys, sample, replace=False)
  return {k: x[k] for k in to_keep}


def int_to_str(k: int) -> str:
  if k % 1000 == 0:
    return str(k//1000) + "k"
  else:
    return str(k)


def val_to_str(val, float_format: str) -> str:
  if val is None:
    return "-"
  if isinstance(val, int):
    return str(val)
  if isinstance(val, str):
    return val
  return float_format % val


def table_string(table: List[List[str]]) -> str:
  """Table as list-of=lists to string."""
  # print while padding each column to the max column length
  if len(table) == 0:
    return ""
  col_lens = [0] * len(table[0])
  for row in table:
    for i, cell in enumerate(row):
      col_lens[i] = max(len(cell), col_lens[i])

  formats = ["{0:<%d}" % x for x in col_lens]
  out = []
  for row in table:
    out.append(" ".join(formats[i].format(row[i]) for i in range(len(row))))
  return "\n".join(out)


def dict_of_dicts_as_table_str(data: Dict[str, Dict[str, Any]], val_format, all_keys=None, top_right="_") -> str:
  """Table of row->col->value to string"""
  if all_keys is None:
    all_keys = {}
    for name, result in data.items():
      for key in result:
        if key not in all_keys:
          all_keys[key] = 0

  all_keys = list(all_keys)
  header = [top_right] + all_keys
  table = [header]
  for name, result in data.items():
    row = [name] + [val_to_str(result.get(x), val_format) for x in all_keys]
    table.append(row)
  return table_string(table)


def list_of_dicts_as_table_str(data: List[Dict[str, Any]], val_format, all_keys=None) -> str:
  """Table of row->col->value to string"""
  if all_keys is None:
    all_keys = {}
    for result in data:
      for key in result:
        if key not in all_keys:
          all_keys[key] = 0

  all_keys = list(all_keys)
  header = all_keys
  table = [header]
  for result in data:
    row = [val_to_str(result.get(x), val_format) for x in all_keys]
    table.append(row)
  return table_string(table)


def nested_struct_to_flat(tensors, prefix=(), cur_dict=None) -> Dict[Tuple, torch.Tensor]:
  if cur_dict is None:
    cur_dict = {}
    nested_struct_to_flat(tensors, (), cur_dict)
    return cur_dict

  if isinstance(tensors, torch.Tensor):
    cur_dict[prefix] = tensors
    return

  if isinstance(tensors, dict):
    if len(tensors) == 0:
      raise ValueError("Cannot convert empty dict")
    for k, v in tensors.items():
      if isinstance(k, int):
        raise NotImplementedError()
      nested_struct_to_flat(v, prefix + (k, ), cur_dict)
  elif isinstance(tensors, (tuple, list)):
    if len(tensors) == 0:
      raise ValueError("Cannot convert empty tuples/lists")
    for ix, v in enumerate(tensors):
      nested_struct_to_flat(v, prefix + (ix, ), cur_dict)
  else:
    raise NotImplementedError()


def flat_to_nested_struct(nested: Dict[Tuple, torch.Tensor]):
  if len(nested) == 0:
    return None
  if isinstance(next(iter(nested.keys()))[0], str):
    out = {}
  else:
    out = []

  for prefix, value in nested.items():
    parent = out
    for i, key in enumerate(prefix[:-1]):
      next_parent = {} if isinstance(prefix[i+1], str) else []
      if isinstance(key, str):
        if key not in parent:
          parent[key] = next_parent
        parent = parent[key]

      elif isinstance(key, int):
        if len(parent) < key + 1:
          parent += [None] * (key + 1 - len(parent))
        if parent[key] is None:
          parent[key] = next_parent
        parent = parent[key]

      else:
        raise NotImplementedError()

    key = prefix[-1]
    if isinstance(key, int):
      if len(parent) < key + 1:
        parent += [None] * (key + 1 - len(parent))
    parent[prefix[-1]] = value

  return out


def process_par(data, fn, n_processes, desc, chunk_size=100, pbar=True, args=[]):
  if n_processes is None or n_processes == 1:
    out = fn(tqdm(data, desc=desc, ncols=100, disable=not pbar), *args)
  else:
    pbar = tqdm(total=len(data), desc=desc, ncols=100, disable=not pbar)
    lock = Lock()

    chunks = split_list(data, n_processes)
    chunks = flatten_list([group_list(c, chunk_size) for c in chunks])

    def call_back(results):
      with lock:
        pbar.update(len(results))

    with Pool(n_processes) as pool:
      results = [
        pool.apply_async(fn, [c] + args, callback=call_back)
        for c in chunks
      ]
      out = None
      for r in results:
        chunk_out = r.get()
        if out is None:
          out = [] if isinstance(chunk_out, list) else {}
        if isinstance(chunk_out, list):
          out += chunk_out
        else:
          out.update(r.get())
    pbar.close()
  return out


def extract_module_from_state_dict(state_dict, name, replace=""):
  name = name + "."
  out = {}
  for k, v in state_dict.items():
    if k.startswith(name):
      out[replace + k[len(name):]] = v
  return out
