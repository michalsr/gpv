import json
from os.path import join, dirname

import h5py
import torch
from typing import Dict, Optional, List, Callable, Union

from allennlp.common import FromParams, Registrable, Params
from allennlp.nn.beam_search import BeamSearch
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from exp.ours.util import our_utils, py_utils
from exp.ours.data.dataset import Dataset
from exp.ours.data.gpv_example import GPVExample
from exp.ours.models.model import GPVModel, GPVExampleOutput, PredictionArg
import numpy as np

from exp.ours.util.load_model import load_model
from exp.ours.util.our_utils import SubsetSampler, DistributedSubsetSampler, QueueDataset
from exp.ours.util.to_params import to_params_any
from utils.io import load_json_object


class GPVExampleOutputHdf5:
  def __init__(self, dataset, text: Optional[List[str]], text_logprobs: Optional[List[float]]):
    self.text = text
    self.text_logprobs = text_logprobs
    self.dataset = dataset
    self._boxes = None
    self._rel = None

  @property
  def boxes(self) -> Optional[np.ndarray]:
    if self.dataset is None:
      return None
    if self._boxes is None:
      self._boxes = self.dataset["boxes"][:]
    return self._boxes

  @property
  def relevance(self) -> Optional[np.ndarray]:
    if self.dataset is None:
      return None
    if self._rel is None:
      self._rel = self.dataset["relevance"][:]
    return self._rel


def save_gpv_output(output: Dict[str, GPVExampleOutput], output_dir):
  predictions = {}
  boxes_h5py = h5py.File(output_dir + "/boxes.hdf5", "w")
  for key, out in output.items():
    grp = boxes_h5py.create_group(str(key))
    if out.boxes is not None:
      grp.create_dataset('boxes', data=out.boxes)
      grp.create_dataset('relevance', data=out.relevance)
    predictions[key] = dict(
      answer=out.text,
      probs=None if out.text_logprobs is None else out.text_logprobs.tolist()
    )

  with open(output_dir + "/predictions.json", "w") as f:
    json.dump(predictions, f)


def load_gpv_predictions(output_dir, load_boxes=False, target_ids=None):
  data = load_json_object(join(output_dir, "predictions.json"))
  if target_ids is not None:
    data = {k: v for k, v in data.items() if k in target_ids}

  if load_boxes:
    f = h5py.File(join(output_dir, "boxes.hdf5"), "r")
    out = {}
    for k, pred in list(data.items()):
      ds = f[k]
      if load_boxes == "hdf5":
        out[k] = GPVExampleOutputHdf5(ds, pred["answer"], pred["probs"])
      else:
        out[k] = GPVExampleOutput(ds["boxes"][:], ds["relevance"][:], pred["answer"], pred["probs"])
    return out
  else:
    return {k: GPVExampleOutput(None, None, v["answer"], v["probs"]) for k, v in data.items()}


def prediction_args_to_json(prediction_args):
  prediction_args_dict = {}
  for k, v in prediction_args.items():
    if isinstance(v, FromParams):
      v = to_params_any(v, Union[PredictionArg, float, int, str])
    prediction_args_dict[k] = v
  return prediction_args_dict


@PredictionArg.register("beam-search-spec")
class BeamSearchSpec(PredictionArg):
  """Specifies how to do a beam search"""

  def __init__(self, beam_size, max_seq_len, per_node_beam_size=None, sampler=None):
    self.beam_size = beam_size
    self.per_node_beam_size = per_node_beam_size
    self.sampler = sampler
    self.max_seq_len = max_seq_len

  def build(self, end_index) -> BeamSearch:
    return BeamSearch(
      end_index, self.max_seq_len, self.beam_size,
      self.per_node_beam_size, self.sampler)


def build_per_example_output(examples, output, beams_to_keep=1):
  out = {}
  for ex, ex_out in zip(examples, output):
    out[ex.id] = ex_out.set_beams_to_keep(beams_to_keep)
  return out


def _run_worker(
    rank, model, devices, queue, out_q, batch_size, num_workers,
    beams_to_keep, prediction_args):
    device = devices[rank]
    with py_utils.DisableLogging():
      model: GPVModel = load_model(model, device=device)
    model.set_prediction_args(**prediction_args)
    dataset = QueueDataset(queue)
    loader = DataLoader(
      dataset, batch_size=batch_size, num_workers=num_workers,
      collate_fn=CollateWithBatch(model.get_collate())
    )
    for examples, batch in loader:
      batch = our_utils.to_device(batch, device)
      with torch.no_grad():
        output = model.predict(**batch)
      out_q.put(build_per_example_output(examples, output, beams_to_keep))


def run(model_source, examples, device,
             batch_size, num_workers, prediction_args, beams_to_keep=1,
             desc="eval", nopbar=False):
  if len(set(ex.get_gpv_id() for ex in examples)) != len(examples):
    raise ValueError("Repeated ids in examples")
  if isinstance(device, list):
    return run_dist(
      model_source, examples, device,
      batch_size, num_workers, prediction_args, beams_to_keep=beams_to_keep,
      desc=desc, nopbar=nopbar
    )
  else:
    if isinstance(model_source, str):
      model = load_model(model_source, device=device)
    data_iterator = DataIterator(batch_size=batch_size, num_workers=num_workers)
    loader = data_iterator.build(model, examples)
    return run_model(model, loader, beams_to_keep, desc, nopbar,
                     prediction_args=prediction_args)


def run_dist(model_source, examples, devices,
             batch_size, num_workers, prediction_args, beams_to_keep=1,
             desc="eval", nopbar=False):
  ctx = torch.multiprocessing.get_context("spawn")

  params = Params(load_json_object(join(dirname(model_source), "model.json")))
  our_utils.import_all()
  with py_utils.DisableLogging():
    model: GPVModel = GPVModel.from_params(params)

  input_q = ctx.Queue()
  for ex in examples:
    # Putting tensors into the queue can lead to tricky errors as it hits
    # torch's tensor-sharing code. It seems easier to just work-around it
    # since we aren't worried about memory consumption anyway
    assert all(not isinstance(v, torch.Tensor) for v in asdict(ex).values())
    # Block so the queue will be completely filled before
    # multiprocessing start
    input_q.put(model.preprocess_example(ex), block=True)

  out_q = ctx.Queue()

  args = (
    model_source, devices, input_q, out_q, batch_size,
    num_workers, beams_to_keep, prediction_args)
  context = torch.multiprocessing.spawn(_run_worker, nprocs=len(devices), args=args, join=False)

  pbar = tqdm(desc=desc, disable=nopbar, total=len(examples), ncols=100)
  out = {}
  while True:
    for k, v in out_q.get().items():
      assert k not in out
      out[k] = v
      pbar.update(1)

    if pbar.n == len(examples):
      assert input_q.empty()
      break
    assert pbar.n < len(examples)

  while not context.join():
    pass

  return out


def run_model(
    model, data_loader, beams_to_keep=1,
    desc="eval", nopbar=False, model_device=None,
    prediction_args=None
) -> Dict[str, GPVExampleOutput]:
  if prediction_args is None:
    prediction_args = {}
  model.eval()
  if model_device is None:
    model_device = our_utils.get_model_device(model)

  model.set_prediction_args(**prediction_args)

  if desc is None:
    desc = "eval"

  out = {}
  if nopbar:
    it = data_loader
  else:
    it = tqdm(data_loader, desc=desc, ncols=100)

  for examples, batch in it:
    batch = our_utils.to_device(batch, model_device)
    with torch.no_grad():
      output = model.predict(**batch)

    out.update(build_per_example_output(examples, output, beams_to_keep))

  return out


# This needs to be a top-level class so it can be distributed
@dataclass
class CollateWithBatch(Callable):
  collate: Callable

  def __call__(self, batch):
    return batch, self.collate(batch)


@dataclass
class DataLoaderBuilder(FromParams):
  batch_size: int
  num_workers: int = 0
  pin_memory: bool = True
  prefetch_factor: int = 2
  persist_workers: bool = False

  def build(self, dataset, collate, sampler=None, shuffle=False,
            batch_size=None, batch_sampler=None):
    return DataLoader(
      dataset,
      batch_size=batch_size if batch_size is not None else self.batch_size,
      collate_fn=collate,
      num_workers=self.num_workers,
      shuffle=shuffle,
      batch_sampler=batch_sampler,
      sampler=sampler,
      pin_memory=self.pin_memory,
      prefetch_factor=self.prefetch_factor,
      persistent_workers=self.persist_workers
    )


@dataclass
class DataIterator(FromParams):
  batch_size: int = 20
  num_workers: int = 8
  sample: int = None
  sort: bool = True
  pin_memory: bool = True

  def build(self, model: GPVModel, examples, world_size=None, rank=None) -> DataLoader:
    if isinstance(examples, Dataset):
      examples = examples.load()
    preprocessed_examples = [model.preprocess_example(x) for x in examples]
    return self.build_from_preprocessed(model.get_collate(), preprocessed_examples, world_size, rank)

  def build_from_preprocessed(self, collate_fn, examples: List[GPVExample], world_size=None, rank=None) -> DataLoader:
    if self.sort:
      if all(ex.sort_len is not None for ex in examples):
        examples.sort(key=lambda x: x.sort_len)

    if world_size is None:
      if self.sample:
        sampler = SubsetSampler(len(examples), self.sample, self.sort)
      else:
        sampler = None
    else:
      sampler = DistributedSubsetSampler(
        len(examples), self.sample, rank, world_size, self.sort)

    _collate = CollateWithBatch(collate_fn)

    return DataLoader(
      examples,
      batch_size=self.batch_size,
      collate_fn=_collate,
      num_workers=self.num_workers,
      shuffle=False,
      sampler=sampler,
      pin_memory=self.pin_memory
    )

