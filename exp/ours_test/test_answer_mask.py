import unittest

import torch
from torch.nn import functional as F
import numpy as np
from transformers import T5Tokenizer

from exp.ours.data.gpv_data import Task
from exp.ours.experimental.answer_masking import AnswerMask, TokenizedWordSearcher, AnswerMaskBuilder


class TestAnswerMasking(unittest.TestCase):

  def test_answer_mask_builder(self):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    builder = AnswerMaskBuilder.build(tokenizer)
    labels = [8, 1782, 3, 16287, 1]  # =tokenizer.encode("the dog jumped")
    mask = builder.build_mask([Task.CLS], torch.as_tensor([labels], dtype=torch.long))
    self.assertEqual(len(mask), 1)
    mask = mask[0]
    self.assertEqual(mask.apply_to.tolist(), [[0, 1]])

    labels = [8, 1782, 3, 16287, 1]  # =tokenizer.encode("the dog jumped")
    mask = builder.build_mask([Task.CLS], torch.as_tensor([labels], dtype=torch.long))
    self.assertEqual(len(mask), 1)
    mask = mask[0]
    self.assertEqual(mask.apply_to.tolist(), [[0, 1]])

  def test_word_searcher(self):
    searcher = TokenizedWordSearcher(
      [np.array(x) for x in [
        [0, 1, 2],
        [5],
        [5, 6, 7],
        [6]
      ]],
      # 6, 7, 8 do not end the previous word
      np.array([True, True, True, True, True, True, False, False, False])
    )
    self.assertEqual(searcher.find(np.array([0, 0, 0, 1, 2, 2, 0])), [(2, 5)])
    self.assertEqual(searcher.find(np.array([0, 0, 0, 1, 2, 2, 5])), [(2, 5), (6, 7)])
    self.assertEqual(searcher.find(np.array([5, 5, 6, 7])), [(0, 1), (1, 4)])
    self.assertEqual(searcher.find(np.array([5, 5, 6, 7, 8])), [(0, 1)])
    self.assertEqual(searcher.find(np.array([5, 8])), [])

  def test_answer_mask(self):
    mask = AnswerMask(
      redistribute_from=torch.as_tensor([2, 5]),
      redistribute_to=torch.as_tensor([0, 1, 3]),
      apply_to=torch.as_tensor([
        [0, 0],
        [0, 1]
      ]),
    )
    src_probs = torch.as_tensor([
      [0.1, 0.1, 0.1, 0.6, 0.1, 0.0],
      [0.15, 0.05, 0.3, 0.1, 0.3, 0.1],
      [0.15, 0.05, 0.3, 0.1, 0.3, 0.1]
    ]).unsqueeze(0)
    out = mask.apply(torch.log(src_probs))
    out_probs = F.softmax(out, -1).numpy()

    # 2 and 5 should be sent to zero
    self.assertTrue(np.allclose(out_probs[:, :2, 2], 0.))
    self.assertTrue(np.allclose(out_probs[:, :2, 5], 0.))

    # 4 should be unchanged
    self.assertTrue(np.allclose(out_probs[:, :2, 4], src_probs[:, :2, 4], rtol=0.0, atol=1e-6))

    # 0, 1, 3 should be increased by the correct constant
    factor1 = 0.9 / (0.9 - 0.1)
    for i in [0, 1, 3]:
      self.assertAlmostEqual(out_probs[0, 0, i], (src_probs[0, 0, i].item()*factor1))

    factor1 = 0.7 / (0.7 - 0.4)
    for i in [0, 1, 3]:
      self.assertAlmostEqual(out_probs[0, 1, i], (src_probs[0, 1, i].item()*factor1))

    # Entry 3 is unchanged
    self.assertAlmostEquals(out_probs[0, 2, i].tolist(), src_probs[0, 2, i].tolist())


if __name__ == '__main__':
  TestAnswerMasking().test_answer_mask_builder()