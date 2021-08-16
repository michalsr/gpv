import logging

from dataclasses import replace

from exp.ours.data.dataset import GpvDataset, Task
from exp.ours.models.t5_custom import OurT5Stack
from exp.ours.models.t5_gpv import T5GPV
from exp.ours.train.runner import BeamSearchSpec
from exp.ours.util import py_utils
from exp.ours.util.load_model import load_model


def test_generate(task: Task, model: T5GPV):
  logging.info("Loading data")
  examples = GpvDataset(task, "val").load()
  examples = examples[:2]

  logging.info("Prep")
  prep = [model.preprocess_example(x) for x in examples]
  prep = [replace(x, query=x.query[:1]) for x in prep]

  inputs = model.get_collate()(prep)
  # inputs["allennlp_spec"] = BeamSearchSpec(1, 30)

  logging.info("Predict1")
  out1 = model(**inputs)


  # t5_model = model.model
  # our_encoder = OurT5Stack(t5_model.encoder.config, t5_model.encoder.embed_tokens)
  # our_encoder.load_state_dict(t5_model.encoder.state_dict())
  #
  # our_decoder = OurT5Stack(t5_model.decoder.config, t5_model.decoder.embed_tokens)
  # our_decoder.load_state_dict(t5_model.decoder.state_dict())
  #
  # t5_model.encoder = our_encoder
  # t5_model.decoder = our_decoder
  #
  # logging.info("Predict2")
  # out2 = model.predict(**inputs)
  #
  # print(out1.text)
  # print(out2.text)


def main():
  py_utils.add_stdout_logger()
  logging.info("Loading model")
  model = load_model("models/t5-cap/lr1e-3-b60-wd1e-4")
  test_generate(Task.CAPTIONING, model)



if __name__ == '__main__':
  main()