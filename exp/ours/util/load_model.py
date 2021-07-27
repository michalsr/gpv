from exp.ours.models.model import GPVModel
from exp.ours.util import py_utils
from exp.ours.util.our_utils import import_all, select_run_dir
import logging
from os.path import dirname, join, exists

import torch
from allennlp.common import Params

from utils.io import load_json_objectz


def load_model(run_dir, use_best_weights=True, device=None,
               quiet=True, epoch=None):
  import_all()
  if run_dir.endswith("/"):
    run_dir = run_dir[:-1]

  run_dir = select_run_dir(run_dir)

  model_spec = join(dirname(run_dir), "model.json")
  params = Params(load_json_object(model_spec))

  with py_utils.DisableLogging():
    model: GPVModel = GPVModel.from_params(params)

  src = None

  if epoch:
    src = join(run_dir, f"state-ep{epoch}.pth")
    if not exists(src):
      raise ValueError(f"Requested epoch {epoch} not found in {run_dir}")

  else:
    if use_best_weights:
      src = join(run_dir, BEST_STATE_NAME)
      if not exists(src):
        src = None
        if not quiet:
          logging.info(f"No best-path found for {run_dir}, using last saved state")

    if src is None:
      raise NotImplementedError()

  if not quiet:
    logging.info("Loading model state from %s" % src)
  # TODO is there way to efficently load the parameters straight to the gpu?
  state_dict = torch.load(src, map_location="cpu")
  model.load_state_dict(state_dict)
  if device is not None:
    model.to(device)
  model.eval()

  # allow state_dict to get freed from memory
  state_dict = None
  del(state_dict)

  return model
