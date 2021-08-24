import unittest
from collections import Counter

from exp.ours.data.stratified_subset_sampler import StratifiedSubsetSampler
import numpy as np

from exp.ours.util import py_utils


class TestStratifiedSubsetSampler(unittest.TestCase):

  def _get_samples(self, group_size, seed, stratify, batch_size, epoch):
    sampler = StratifiedSubsetSampler(group_size, seed=seed, stratify=stratify, batch_size=batch_size)
    sampler.set_epoch(epoch)
    if batch_size is None:
      return list(sampler)
    else:
      batches = list(sampler)
      self.assertTrue(all(len(x) <= batch_size for x in batches))
      return py_utils.flatten_list(batches)

  def test_basic(self):
    for batch in [None, 7]:
      for group_size in [[20], [20, 10], [50, 60, 30]]:
        for stratify in [True, False]:
          seed = np.random.randint(0, 10000)
          samples0 = self._get_samples(group_size, seed, stratify, batch, 0)
          self.assertEqual(set(samples0), set(range(sum(group_size))))

          # Should be repeatable
          samples = self._get_samples(group_size, seed, stratify, batch, 0)
          self.assertEqual(samples, samples0)

          # Should be different with different seed
          samples = self._get_samples(group_size, seed+1, stratify, batch, 0)
          self.assertNotEqual(samples0, samples)

          # And with different epochs
          samples = self._get_samples(group_size, seed, stratify, batch, np.random.randint(1, 1000))
          self.assertNotEqual(samples0, list(samples))

  def test_sample_counts(self):
    sampler = StratifiedSubsetSampler(
      [20, 8, 15], samples_per_epoch=[5, None, 0.333], seed=0, stratify=True)
    self.assertEqual(len(sampler), 5+8+5)
    for epoch in [0, 3, 5, 12]:
      sampler.set_epoch(epoch)
      points = list(sampler)
      self.assertEqual(sum(x < 20 for x in points), 5)
      self.assertEqual(sum(20 <= x < 28 for x in points), 8)
      self.assertEqual(sum(28 <= x for x in points), 5)

  def test_cycle_exact(self):
    sampler = StratifiedSubsetSampler(
      [20], samples_per_epoch=[10], seed=0, stratify=True)

    for offset in [0, 8]:
      sampler.set_epoch(offset + 0)
      epoch0 = list(sampler)
      sampler.set_epoch(offset + 1)
      epoch1 = list(sampler)
      self.assertEqual(len(epoch0), 10)
      self.assertEqual(len(epoch1), 10)
      self.assertEqual(set(epoch0 + epoch1), set(range(20)))

  def test_no_repetas(self):
    for n, n_epochs in [(2, 10), (8, 3), (14, 2)]:
      sampler = StratifiedSubsetSampler(
        [29], samples_per_epoch=[n], seed=1, stratify=True)
      all_seen = set()
      for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        all_seen.update(sampler)
      self.assertEqual(len(all_seen), len(set(all_seen)))

  def test_cycles(self):
    for grp_sz, samples in [
      ([5], [4]),
      ([5], [2]),
      ([17], [11]),
      ([21, 23, 18], [10, 19, 6])
    ]:
      sampler = StratifiedSubsetSampler(
        grp_sz, samples_per_epoch=samples, seed=0, stratify=True)
      bounds = np.cumsum(sampler.group_sizes)
      counts = Counter()
      for epoch in range(12):
        sampler.set_epoch(epoch)
        counts.update(list(sampler))

      for i in range(len(bounds)):
        start = 0 if i == 0 else bounds[i-1]
        end = bounds[i]
        grp = [counts[x] for x in range(start, end)]
        # Even sampling should always ensure examples appear in equal proportions
        self.assertLessEqual(max(grp)-min(grp), 1)

  def test_distributed(self):
    word_size = 4
    sizes = [17, 13, 27, 37]
    samples = [4, 10, 20, None]
    batch_size = 3
    for epoch in [0, 12]:
      distributed_batches = []
      for rank in range(word_size):
        sampler = StratifiedSubsetSampler(
          sizes, samples_per_epoch=samples, seed=9213, stratify=True,
          rank=rank, world_size=word_size, batch_size=batch_size)
        sampler.set_epoch(epoch)
        distributed_batches.append(list(sampler))
        self.assertEqual(len(sampler), len(distributed_batches[-1]))

      self.assertTrue(all(len(distributed_batches[0]) == len(x) for x in distributed_batches[1:]))
      all_distributed = py_utils.flatten_list(py_utils.flatten_list(distributed_batches))

      sampler = StratifiedSubsetSampler(
        sizes, samples_per_epoch=samples, seed=9213, stratify=True)
      sampler.set_epoch(epoch)
      self.assertEqual(set(all_distributed), set(sampler))


if __name__ == '__main__':
  TestStratifiedSubsetSampler().test_distributed()
  # unittest.main()