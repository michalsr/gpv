import unittest

from transformers import AutoTokenizer

import torchvision.transforms as T

from exp.ours.data.gpv_example import GPVExample
from exp.ours.data.dataset import Task
from exp.ours.models.t5_gpv import T5Collate
from exp.ours.util.image_utils import DUMMY_IMAGE_ID
from exp.ours.util.nlp_utils import encode_with_cache
import numpy as np


class TestT5GPV(unittest.TestCase):

  def test_collate(self):
    test_query = [
      'What is happening in this image.',
      'Generate a description for the image.',
    ]
    test_answers = [
      'A giraffe at a zoo enjoying the day',
      'A patterned bowl full of oranges sitting on a colorful background'
    ]

    trans = T.Compose([
      T.ToPILImage(mode='RGB'),
      T.ToTensor()
    ])

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    cache = {}

    enc_q = [np.array(encode_with_cache(q, tokenizer, cache)) for q in test_query]
    enc_a = [np.array(encode_with_cache(a, tokenizer, cache)) for a in test_answers]

    our_collate = T5Collate(
      True, tokenizer, (480, 600),
      {t: trans for t in Task}, trans, 512, 512, pre_tokenized=True)
    our_out = our_collate(
      [GPVExample(
        str(i), Task.CAPTIONING, DUMMY_IMAGE_ID, [q], target_answer=a
      ) for i, (q, a) in enumerate(zip(enc_q, enc_a))]
    )
    ours = our_out

    expected_q = tokenizer(test_query, return_tensors='pt', padding=True)
    expected_a = tokenizer(test_answers, return_tensors='pt', padding=True)

    self.assertEqual(ours["input_ids"].tolist(), expected_q["input_ids"].tolist())
    self.assertEqual(ours["output_ids"].tolist(), expected_a["input_ids"].tolist())
    self.assertEqual(ours["input_mask"].tolist(), expected_q["attention_mask"].tolist())
    self.assertEqual(ours["output_mask"].tolist(), expected_a["attention_mask"].tolist())


if __name__ == '__main__':
  unittest.main()
