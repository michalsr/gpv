import logging
from typing import List

from allennlp.common import Registrable
from torch.utils.data import Dataset as TorcDataset

from exp.ours.data.dataset import Task
from exp.ours.data.gpv_example import GPVExample
import numpy as np


class DatasetBuilder(Registrable):

  def build(self, examples: List[GPVExample], seed) -> TorcDataset:
    # Note this might be run on multiple processes, so it should be deterministic,
    raise NotImplementedError()


@DatasetBuilder.register("partition-web-qa")
class PartitionWebQa(DatasetBuilder):

  def __init__(self, n_partitions: int):
    self.n_partitions = n_partitions

  def build(self, examples: List[GPVExample], seed) -> TorcDataset:
    always_keep = []
    webqa = []
    for ex in examples:
      if ex.task == Task.WEBQA:
        webqa.append(ex)
      else:
        always_keep.append(ex)
    if len(webqa) == 0:
      return always_keep
    keep = len(webqa) // self.n_partitions
    logging.info(f"Keeping {len(always_keep)} examples and sampling {keep} of {len(webqa)} from webqa")
    return SubsetDataset(always_keep, webqa, self.n_partitions, seed)


# TODO rename to `PartitionedSubsetDataset`
class SubsetDataset(TorcDataset):
  def __init__(self, always_include: List, sample_partition: List,
               partitions: int, seed: int):
    self.always_include = always_include
    self.n_partitions = partitions
    self.sample_partition = list(sample_partition)
    self.seed = seed

    partition_size = len(sample_partition)//partitions
    partition_sizes = [partition_size] * partitions
    for r in range(len(sample_partition) - partition_size*partitions):
      partition_sizes[r] += 1
    assert sum(partition_sizes) == len(sample_partition)
    self._bounds = np.cumsum([0] + partition_sizes)

    self._on_partition = None
    self.size = len(sample_partition)

  def __len__(self):
    start, end = self._bounds[self._on_partition], self._bounds[self._on_partition+1]
    return len(self.always_include) + end - start

  def __getitem__(self, ix):
    if ix < len(self.always_include):
      return self.always_include[ix]
    ix -= len(self.always_include)
    start = self._bounds[self._on_partition]
    return self.sample_partition[start + ix]

  def set_epoch(self, epoch):
    if self._on_partition is None or self._on_partition == self.n_partitions - 1:
      rng = np.random.RandomState(self.seed + epoch*4523)
      self._on_partition = 0
      rng.shuffle(self.sample_partition)
    else:
      self._on_partition = self._on_partition + 1
