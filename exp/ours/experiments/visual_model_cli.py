from argparse import ArgumentParser
from os.path import exists

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from exp.ours.image_featurizer.clip_featurizer import ClipFeaturizer, RoIFeatureSource
#from exp.ours.image_featurizer.detectron_detectors import DetectronBackboneWithBoxes
from exp.ours.image_featurizer.detr_featurizer import PretrainedDetrFeaturizer, \
  BoxesWithDetrBackbone
from exp.ours.image_featurizer.image_featurizer import *
from exp.ours.image_featurizer import vinvl_featurizer
from exp.ours.image_featurizer import detr_featurizer
from exp.ours.image_featurizer.vinvl_featurizer import *
from exp.ours.models.layers import *


def add_image_featurizer_args(parser: ArgumentParser, vfreeze="none", vmodel=None):
  parser.add_argument("--vmodel", default=vmodel)
  parser.add_argument("--vfreeze", default=vfreeze)


def _freeze_detr(vfreeze):
  freeze_extractor = False
  if vfreeze is None or vfreeze == "none":
    freeze_backbone = None
  elif vfreeze == "conv1":
    freeze_backbone = "conv1"
  else:
    freeze_backbone = "all"
    if vfreeze == "backbone":
      freeze_extractor = False
    elif vfreeze == "all":
      freeze_extractor = True
    else:
      raise NotImplementedError()
  return freeze_backbone, freeze_extractor


def get_image_featurizer(args) -> Tuple[ImageFeatureExtractor, int]:
  """Returns """

  if args.vmodel == "detr_model":
    freeze_backbone, freeze_extractor = _freeze_detr(args.vfreeze)
    dim = 2304
    extractor = detr_featurizer.PretrainedDetrFeaturizer(
      freeze_backbone=freeze_backbone, freeze_extractor=freeze_extractor, pretrained_model="coco")

  elif args.vmodel in {"dbg", "debug"}:
    dim = 2048+5
    extractor = DebugFeaturizer(50, 2048+5)

  elif args.vmodel == "clip":
    # dim = 1285
    # extractor = ClipFeaturizer(
    #   "RN50x4", "vinvl", [
    #     RoIFeatureSource(1, (7, 7,)),
    #     RoIFeatureSource(2, (3, 3,)),
    #     RoIFeatureSource(3, (1, 1,)),
    #   ],
    #   freeze="all",
    #   coordinate_embedder=BasicBoxEmbedder(),
    # )
    #
    dim = 10240//4*9 + 5
    extractor = ClipFeaturizer(
      "RN50x4", "vinvl", [
        # RoIFeatureSource(1, (3, 3,), aggregate="flatten"),
        # RoIFeatureSource(2, (2, 2,), aggregate="flatten"),
        RoIFeatureSource(3, (3, 3,), aggregate="flatten"),
        # RoIFeatureSource(3, (3, 3,), aggregate="flatten"),
      ],
      freeze="all",
      coordinate_embedder=BasicBoxEmbedder(),
    )

  elif args.vmodel == "vinvl-precomputed":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2053
    extractor = vinvl_featurizer.VinVLPrecomputedFeatures()
  elif args.vmodel == "dbg-hdf5":
    dim = 2048
    extractor = Hdf5FeatureExtractor("dbg")
  elif args.vmodel == "dbg-hdf5":
    dim = 2048
    extractor = Hdf5FeatureExtractor("dbg")
  elif args.vmodel in {"vinvl", "vinvl-web"}:  # winvl-web
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl", box_embedder=BasicBoxEmbedder())
  elif args.vmodel == "vboxes-detr":
    dim = 2048 + 5
    extractor = BoxesWithDetrBackbone(
      "vinvl", "coco",
      feature_extractor=BoBoxEmbedFeatureExtractor(
        box_coordinate_embed=BasicBoxEmbedder(),
        post_rio=LayerNorm(),
      ),
    )
  else:
    raise NotImplementedError(args.vmodel)
  return extractor, dim