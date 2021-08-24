from typing import List, Union, Optional

from torch.utils.data import Dataset as TorchDataset, Sampler

import numpy as np

from exp.ours.util import py_utils
from exp.ours.util.our_utils import get_batch_bounds


class StratifiedSubsetSampler(Sampler):
  """Sampler that supports stratifying different groups, distributed training, and
  showing different subsets of each group each epoch

  Requires `set_epoch` to be called before each epoch
  """

  def __init__(
      self,
      group_sizes: List[int],
      seed: Optional[int],
      stratify: bool,
      samples_per_epoch: List[Union[None, int, float]]=None,
      rank=None,
      world_size=None
  ):
    self.group_sizes = group_sizes
    if samples_per_epoch is None:
      samples_per_epoch = [None for _ in group_sizes]
    self.samples_per_epoch = samples_per_epoch
    self.seed = seed
    self.stratify = stratify
    self.rank = rank
    self.world_size = world_size

    self._epoch_data = []
    self._size_per_epoch = []
    self._group_offests = np.cumsum(group_sizes)
    for i, val in enumerate(samples_per_epoch):
      if val is None:
        self._size_per_epoch.append(group_sizes[i])
      elif isinstance(val, float):
        assert 0 < val <= 1.0
        self._size_per_epoch.append(int(round(group_sizes[i]*val)))
      else:
        assert val <= group_sizes[i]
        self._size_per_epoch.append(val)

    total_examples = sum(group_sizes)
    if world_size is not None:
      self.bounds = get_batch_bounds(total_examples, self.world_size)
    else:
      self.bounds = None

    self.n = sum(self._size_per_epoch)

  def _get_seed(self, group_cycle, ix):
    return self.seed + group_cycle*13 + 2039 + ix*17

  def set_epoch(self, epoch):
    # First decide which indices to include in this epoch
    all_data = []
    for i, group_sz in enumerate(self.group_sizes):
      offset = 0 if i == 0 else self._group_offests[i-1]
      sz = self._size_per_epoch[i]
      start = sz * epoch % group_sz
      end = start + sz
      group_cycle = ((epoch + 1)*sz - 1) // group_sz
      group_rng = np.random.RandomState(self._get_seed(group_cycle, i))
      indices = group_rng.permutation(group_sz)

      if end <= group_sz:
        # No wrap around
        all_data.append(indices[start:end] + offset)
      else:
        # Wraps around, get data from the previous cycle
        all_data.append(indices[:(end % group_sz)] + offset)
        group_rng = np.random.RandomState(self._get_seed(group_cycle-1, i))
        indices = group_rng.permutation(group_sz)
        all_data.append(indices[start:] + offset)

    # The shuffle them and optionally merge them in a stratified way
    shuffle_rng = np.random.RandomState(self.seed + 	5417)
    if self.stratify:
      for grp in all_data:
        shuffle_rng.shuffle(grp)
      self._epoch_data = py_utils.balanced_merge_multi(all_data)
    else:
      self._epoch_data = py_utils.flatten_list(all_data)
      shuffle_rng.shuffle(self._epoch_data)

  def __iter__(self):
    if self.world_size is not None:
      s, e = self.bounds[self.rank]
      return iter(self._epoch_data[s:e])
    else:
      return iter(self._epoch_data)

  def __len__(self):
    return self.n

