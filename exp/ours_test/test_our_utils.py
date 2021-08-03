import unittest

import torch

from exp.ours.util import our_utils


class TestOurUtils(unittest.TestCase):

  def test_concat_masked_seq(self):
    for _ in range(10):
      batch_size = 3
      dim = 2
      seq_len1 = torch.randint(0, 8, (batch_size,))
      seq_len2 = torch.randint(0, 8, (batch_size,))
      seq1 = torch.empty(batch_size, seq_len1.max(), dim).uniform_(-1, 1)
      seq2 = torch.empty(batch_size, seq_len2.max(), dim).uniform_(-1, 1)
      out, out_mask = our_utils.concat_masked_sequences(seq1, seq_len1, seq2, seq_len2)
      assert out.size(1) == out_mask.size(1)

      for i in range(batch_size):
        for z in range(seq_len1[i]):
          self.assertEquals(out[i, z].tolist(), seq1[i, z].tolist())
          self.assertTrue(out_mask[i, z])
        for z in range(seq_len2[i]):
          self.assertEquals(out[i, z + seq_len1[i]].tolist(), seq2[i, z].tolist())
          self.assertTrue(out_mask[i, z + seq_len1[i]])
        for z in range(seq_len1[i]+seq_len2[i], out.size(1)):
          self.assertFalse(out_mask[i, z])

  def test_build_vocab_mask(self):
    labels = torch.as_tensor([
      [0, 5, 1, 0, 4],
      [0, 5, 2, 3, 4],
    ])
    to_mask = [
      [[1], [0, 2], [5, 4], [6]],
      [[5, 1], [3, 5], [6, 2]]
    ]
    mask = our_utils.build_vocab_mask(labels, to_mask, 7)
    expected = torch.zeros_like(mask[0])
    expected[:, 1] = True
    expected[:, 6] = True
    expected[0, 2] = True
    expected[3, 2] = True
    expected[1, 4] = True
    self.assertTrue(torch.all(mask[0] == expected))

    expected = torch.zeros_like(mask[0])
    expected[1, 1] = True
    expected[3, 5] = True
    self.assertTrue(torch.all(mask[1] == expected))


if __name__ == '__main__':
  TestOurUtils().test_concat_masked_seq()
