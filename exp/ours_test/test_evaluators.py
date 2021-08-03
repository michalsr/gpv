import unittest

from exp.ours.train.runner import GPVExampleOutput
from exp.ours.data.source_data import CocoCaptions, CocoCaption
from exp.ours.train.evaluator import CaptionEvaluator, ResultKey


EXAMPLE1 = CocoCaptions(
  image_id=12902,
  captions=[
    CocoCaption(id=51, caption='A black and white bowl of oranges on a red and purple background.'),
    CocoCaption(id=111, caption='A bowl of large oranges in a designer bowl.'),
    CocoCaption(id=123, caption='An abstract designed bowl holding a bunch of oranges.'),
    CocoCaption(id=165, caption='A patterned bowl full of oranges sitting on a colorful background'),
    CocoCaption(id=252, caption='A colorful picture of oranges in a bowl.')
  ]
)
EXAMPLE2 = CocoCaptions(
  image_id=12057,
  captions=[
    CocoCaption(id=112, caption='A giraffe at a zoo enjoying the day'),
    CocoCaption(id=175, caption='An adult giraffe basking in the shade at a zoo'),
    CocoCaption(id=4507, caption='A giraffe standing in a pen in the shade.'),
    CocoCaption(id=6361, caption='A giraffe is walking outdoors in its inclosure.'),
    CocoCaption(id=6760, caption='A giraffe standing inside its cage in the shade')
  ]
)


class TestEvaluators(unittest.TestCase):

  def test_caption(self):
    for i in range(3):
      out = CaptionEvaluator(caching_tokenizer=True).evaluate(
        [EXAMPLE1, EXAMPLE2],
        {
          EXAMPLE1.get_gpv_id(): GPVExampleOutput(None, None,
                                                  ['a bunch of oranges in a glass bowl'.upper()], None),
          EXAMPLE2.get_gpv_id(): GPVExampleOutput(None, None,
                                                  ['a giraffe walking in a fenced in area.'], None)
        },
        allow_partial=True
      )
      for (k, expected) in [
        ('bleu1', 0.8124999998984376),
        ('bleu2', 0.6373774391165805),
        ('bleu3', 0.5135200122431356),
        ('bleu4', 0.40567246283722375),
        ('cider', 1.657245555)
      ]:
        self.assertAlmostEqual(out[ResultKey(k)], expected, places=5)


if __name__ == '__main__':
  unittest.main()
