import logging
import os
from collections import defaultdict
from os import listdir, path, walk
from os.path import join, exists, isdir, basename, relpath, dirname
from typing import List, Union, Optional

import numpy as np
import torch
from allennlp.common.util import import_module_and_submodules
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Sampler, IterableDataset

from exp.gpv.models.detr_roi_head import create_detr_roi_head
from exp.ours import file_paths
from utils.io import load_json_object
import torch.nn.functional as F
IMPORT_DONE = False


def import_all():
  global IMPORT_DONE
  if not IMPORT_DONE:
    for k in ["models", "train", "image_featurizer"]:
      import_module_and_submodules(f"exp.ours.{k}")
    IMPORT_DONE = True


def get_device(device_name: Union[None, str, int]=None):
  if device_name is None:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda")
      return torch.device('cuda')
    else:
      logging.info("cuda not found, using cpu")
      return torch.device('cpu')
  else:
    try:
      device_name = int(device_name)
    except ValueError:
        pass
    return torch.device(device_name)


def get_devices(devices):
  if devices is None and "PYTORCH_DEVICES" in os.environ:
    devices = os.environ["PYTORCH_DEVICES"].split()

  if isinstance(devices, list) and len(devices) > 1:
    out = []
    for x in devices:
      try:
        out.append(int(x))
      except ValueError:
        out.append(x)
    return out

  if isinstance(devices, list):
    devices = devices[0]

  if devices is not None:
    try:
      return int(devices)
    except ValueError:
      return devices
  else:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda")
      return 'cuda'
    else:
      logging.info("cuda not found, using cpu")
      return 'cpu'



def replace_params_with_buffers(module: nn.Module):
  for m in module.modules():
    for n, param in list(m.named_parameters(recurse=False)):
      del m._parameters[n]
      module.register_buffer(n, param.data)


def is_model_dir(x):
  return exists(join(x, "model.json"))


def is_run_dir(x, require_done):
  if exists(join(x, "status.json")):
    if not require_done:
      return True
    else:
      return load_json_object(join(x, "status.json"))["done"]
  return False


def extract_runs(model_dir, require_done=True):
  runs = []
  for run_dir in listdir(model_dir):
    run_dir = join(model_dir, run_dir)
    if is_run_dir(run_dir, require_done):
      runs.append(run_dir)
  return runs


def stack_and_pad(tensors: List, max_len=None, pad=0.0) -> torch.Tensor:
  tensors = [torch.as_tensor(x) for x in tensors]
  if max_len is None:
    max_len = max(x.size(0) for x in tensors)
  t0 = tensors[0]
  if len(t0.size()) == 2:
    out = torch.full((len(tensors), max_len, t0.size(1)), pad, dtype=t0.dtype, device=t0.device)
  else:
    out = torch.full((len(tensors), max_len), pad, dtype=t0.dtype, device=t0.device)

  for i, t in enumerate(tensors):
    out[i, :t.size(0)] = t
  return out


def binary_mask(seq_lens, max_len=None):
  if max_len is None:
    max_len = seq_lens.max()
  ixs = torch.arange(max_len, device=seq_lens.device, dtype=seq_lens.dtype)
  return ixs.unsqueeze(0) < seq_lens.unsqueeze(0)


def find_models(roots, require_runs=True, require_done=True):
  if isinstance(roots, str) and is_run_dir(roots, require_done):
    return {path.split(roots)[1]: (dirname(roots), [roots])}

  if isinstance(roots, str):
    roots = [(None, roots)]
  elif isinstance(roots, dict):
    roots = list(roots.items())
  elif len(roots) == 1:
    roots = [(None, roots[0])]
  else:
    names = [x.rstrip("/").split("/")[-2] for x in roots]
    roots = list(zip(names, roots))

  models = {}
  for root_name, root in roots:
    if is_model_dir(root):
      runs = []
      for run_dir in listdir(root):
        run_dir = join(root, run_dir)
        if is_run_dir(run_dir, require_done):
          runs.append(run_dir)
      model_name = basename(root)
      if root_name:
        model_name = join(root_name, model_name)
      models[model_name] = (root, runs)
      continue

    for dirpath, dirnames, filenames in walk(root):
      for model_dir in dirnames:
        model_dir = join(dirpath, model_dir)
        if not is_model_dir(model_dir):
          continue

        model_name = relpath(model_dir, root)
        if root_name:
          model_name = join(root_name, model_name)

        runs = extract_runs(model_dir, require_done)
        if not require_runs or len(runs) > 0:
          models[model_name] = (model_dir, runs)

  return models


def load_gpv1(config, state, loc):
  from exp.gpv.models.gpv import GPV
  model = GPV(config)
  model.to(loc)
  loaded_dict = torch.load(state, map_location=loc)['model']
  state_dict = model.state_dict()
  for k, v in state_dict.items():
    state_dict[k] = loaded_dict[f'module.{k}']

  model.load_state_dict(state_dict)
  model.eval()
  return model


def to_device(batch, device):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: to_device(sub_v, device) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [to_device(x, device) for x in batch]
  else:
    return batch.to(device)


def pin_memory_recursive(batch):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: pin_memory_recursive(sub_v) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [pin_memory_recursive(x) for x in batch]
  else:
    return batch.pin_memory()


def get_model_device(module: torch.nn.Module):
  return next(module.parameters()).device


def seq_len_to_binary_mask(seq_len, max_len=None):
  if max_len is None:
    max_len = seq_len.max()
  return seq_len.unsqueeze(1) > torch.arange(0, max_len, device=seq_len.device).unsqueeze(0)


def concat_masked_sequences(
    seq1, mask1,
    seq2, mask2
):
  batch = seq1.size(0)
  if mask1 is None and mask2 is None:
    return torch.cat([seq1, seq2], 1), None
  if mask1 is None:
    if len(mask2.size()) == 1:
      raise NotImplementedError("Sequence length masks2")
    out = torch.cat([seq1, seq2], 1)
    mask = torch.cat([
      torch.ones(batch, seq1.size(1), device=seq1.device, dtype=mask2.dtype),
      mask2
    ], 1)
    return out, mask
  elif mask2 is None:
    seq2_len = seq2.size(1)

    if len(mask1.size()) == 2:
      assert mask1.dtype == torch.bool or torch.all(torch.logical_or(mask1 == 0, mask1 == 1))
      seq_len1 = mask1.int().sum(1)
    else:
      assert mask1.dtype == torch.long and len(mask1.size()) == 1
      seq_len1 = mask1

    out = F.pad(seq1, [0, 0, 0, seq2_len, 0, 0])
    for i in range(batch):
        out[i, seq_len1[i]:seq_len1[i]+seq2_len] = seq2[i]
    return out, seq_len_to_binary_mask(seq_len1 + seq2_len)
  else:
    # both mask are not none
    if len(mask1.size()) != 1:
      raise NotImplementedError("Binary mask1")
    else:
      seq_len1 = mask1

    if len(mask2.size()) == 2:
      assert mask2.dtype == torch.bool or torch.all(torch.logical_or(mask2 == 0, mask2 == 1))
      seq_len2 = mask2.int().sum(1)
    else:
      seq_len2 = mask2

    out_len = (seq_len1 + seq_len2).max()
    to_pad = out_len - seq1.size(1)
    out = F.pad(seq1, [0, 0, 0, to_pad, 0, 0])
    for i in range(batch):
      out[i, seq_len1[i]:seq_len1[i]+seq_len2[i]] = seq2[i, :seq_len2[i]]
    return out, seq_len_to_binary_mask(seq_len1 + seq_len2)


def get_batch_bounds(n, n_batches):
  per_group = n // n_batches
  remainder = n % n_batches
  goup_sizes = np.full(n_batches, per_group, np.int)
  goup_sizes[:remainder] += 1

  batch_ends = np.cumsum(goup_sizes)

  assert batch_ends[-1] == n

  batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")
  bounds = np.stack([batch_starts, batch_ends], 1)
  return bounds


class QueueDataset(IterableDataset):
  def __init__(self, q):
    self.q = q

  def __iter__(self):
    while True:
      element = self.q.get()
      if element is None:
        return
      yield element


class SubsetSampler(Sampler):
  def __init__(self, n: int, n_examples: int, sort=False):
    if not isinstance(n, int) or not isinstance(n_examples, int):
      raise ValueError("args should be integers")
    if n_examples > n:
      raise ValueError(f"Requested {n_examples} samples, but only {n} examples.")
    self.n = n
    self.sort = sort
    self.n_examples = n_examples

  def __iter__(self):
    ixs = np.random.choice(self.n, self.n_examples, replace=False)
    if self.sort:
      ixs.sort()
    return iter(ixs)

  def __len__(self):
    return self.n_examples


class DistributedSubsetSampler(Sampler):
  def __init__(self, n: int, n_examples: Optional[int], rank: int,
               world_size: int, sort=False, seed=0):
    if not isinstance(n, int):
      raise ValueError("args should be integers")
    if n_examples is not None and n_examples > n:
      raise ValueError(f"Requested {n_examples} examples, but only have {n} total")
    self.n = n
    self.rank = rank
    self.sort = sort
    self.world_size = world_size
    self.n_examples = n_examples
    self.seed = seed
    self.bound = get_batch_bounds(n if n_examples is None else n_examples, self.world_size)

  def set_seed(self, seed):
    self.seed = seed

  def __iter__(self):
    g = torch.Generator()
    g.manual_seed(self.seed)
    indices = torch.randperm(self.n, generator=g)
    s, e = self.bound[self.rank]
    subset = indices[s:e].tolist()
    if self.sort:
      subset.sort()
    return iter(subset)

  def __len__(self):
    s, e = self.bound[self.rank]
    return e - s


def select_run_dir(run_dir):
  if exists(join(run_dir, "model.json")):
    candidates = []
    for filename in listdir(run_dir):
      filepath = join(run_dir, filename)
      if isdir(filepath) and filename.startswith("r"):
        candidates.append(filepath)
    if len(candidates) > 1:
      raise ValueError(f"Multiple runs in {run_dir}, please select one")
    elif len(candidates) == 0:
      raise ValueError(f"No runs found in {run_dir}")
    else:
      logging.info(f"Selecting run {basename(candidates[0])} for {run_dir}")
      run_dir = candidates[0]

  return run_dir


def get_detr_model(num_queries=100, pretrained="coco_sce", lr_backbone=0,
                   load_object_classifier=False):
  cfg = dict(
    num_queries=num_queries,
    num_classes=91 if load_object_classifier else 1,
    hidden_dim=256,
    lr_backbone=lr_backbone,
    nheads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    backbone="resnet50",
    position_embedding="sine",
    masks=False,
    dilation=False,
    dropout=0.1,
    dim_feedforward=2048,
    pre_norm=False,
    aux_loss=False,
    frozenbatchnorm=True,
    last_layer_only=True,
  )
  model = create_detr_roi_head(OmegaConf.create(cfg))

  if pretrained:
    state_dict = torch.load(
      file_paths.PRETRAINED_DETR_MODELS[pretrained], map_location="cpu")['model']
    if not load_object_classifier:
      del state_dict["class_embed.weight"]
      del state_dict["class_embed.bias"]
      model.load_state_dict(state_dict, strict=False)
    else:
      model.load_state_dict(state_dict)

  return model


def build_vocab_mask(labels, to_mask: List[List[List[int]]], voc_size, eos_id=None):
  batch, seq_len = labels.size()
  out = torch.zeros(batch, seq_len, voc_size, dtype=torch.bool)

  for batch_ix, example_mask in enumerate(to_mask):
    second_to_last_to_mask = defaultdict(list)
    for mask in example_mask:
      if len(mask) == 1:
        out[batch_ix, :, mask[0]] = True
      else:
        second_to_last_to_mask[mask[-2]].append(mask)
    if len(second_to_last_to_mask) == 0:
      continue
    for seq_ix in range(seq_len):
      token_id = labels[batch_ix, seq_ix]
      if token_id == eos_id:
        break
      for candidate in second_to_last_to_mask[token_id.item()]:
        if len(candidate) == 2:
          out[batch_ix, seq_ix, candidate[1]] = True
        elif len(candidate) > seq_ix:
          raise None
  return out

