import logging
from os.path import join, dirname

from . import defaults
from . import sg_defaults
from .attr_rcnn import AttrRCNN
from .structures.image_list import to_image_list
from .transforms import build_transforms
from .utils.checkpoint import DetectronCheckpointer
from ..ours import config
from ..ours.util.py_utils import DisableLogging



def vinvl_colllate_images(batch):
  transposed_batch = list(zip(*batch))
  images = to_image_list(transposed_batch[0])
  targets = transposed_batch[1]
  img_ids = transposed_batch[2]
  return images, targets, img_ids


def get_vinvl(model="release"):
  if model == "release":
    src = join(config.VINVL_SOURCE, "release")
  elif model == "R50C4_4sets":
    src = join(config.VINVL_SOURCE, "R50C4_4sets_174000_model.roi_heads.score_thresh_0.2")
  elif model == "R50C4_4setsvg":
    src = join(config.VINVL_SOURCE, "R50C4_4setsvg_005000_model.roi_heads.score_thresh_0.2")
  else:
    raise NotImplementedError()

  state = join(src, "model.pth")
  model_cfg = join(src, "config.yaml")

  logging.info(f"Initializing VinVL model")
  cfg = defaults._C.clone()
  cfg.set_new_allowed(True)
  cfg.merge_from_other_cfg(sg_defaults._C)

  cfg.merge_from_file(model_cfg)
  cfg.set_new_allowed(False)

  # Parameter recommended for generating features
  cfg.TEST.OUTPUT_FEATURE = True
  cfg.MODEL.ATTRIBUTE_ON = False
  # TODO should we be using the regression predictions for Localization?
  cfg.TEST.IGNORE_BOX_REGRESSION = True
  cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.2
  cfg.MODEL.ROI_HEADS.NMS_FILTER = 1
  cfg.MODEL.ROI_BOX_HEAD.COMPUTE_BOX_LOSS = True

  cfg.freeze()
  model = AttrRCNN(cfg)

  eval_transform = build_transforms(cfg, is_train=False)

  logging.info(f"Loading vinvl state from {state}")
  checkpointer = DetectronCheckpointer(cfg, model)
  with DisableLogging():
    checkpointer.load(state)
  model.device = None
  model.eval()
  return model, eval_transform
