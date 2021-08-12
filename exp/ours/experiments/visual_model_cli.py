from argparse import ArgumentParser

from exp.ours.image_featurizer.image_featurizer import *
from exp.ours.image_featurizer import vinvl_featurizer
from exp.ours.image_featurizer import detr_featurizer
from exp.ours.image_featurizer.vinvl_featurizer import VinvlBackboneImageFeaturizer
from exp.ours.models.layers import DetrBackbone, LayerNorm, BasicBoxEmbedder, \
  NonLinearCoordinateEncoder


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
      freeze_backbone=freeze_backbone, freeze_extractor=freeze_extractor)
  elif args.vmodel == "faster_rcnn":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("faster-rcnn", "xyxy", BasicBoxEmbedder())
  elif args.vmodel in {"dbg", "debug"}:
    dim = 32
    extractor = DebugFeaturizer(4, 32)
  elif args.vmodel == "vinvl-precomputed":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2054
    extractor = vinvl_featurizer.VinVLPrecomputedFeatures()
  elif args.vmodel == "dbg-hdf5":
    dim = 2048
    extractor = Hdf5FeatureExtractor("dbg")
  elif args.vmodel == "vinvl":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl", box_embedder=BasicBoxEmbedder())
  elif args.vmodel == "web-vinvl":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor(["vinvl", "web-features"], box_embedder=BasicBoxEmbedder())
  elif args.vmodel == "vinvl-r50c4-4setvg":
    if args.vfreeze != "all":
      raise ValueError()
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl-r50c4-4setvg", "xyxy", BasicBoxEmbedder())
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes":
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl-r50c4-4setvg-rboxes", box_embedder=BasicBoxEmbedder())
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes-all-image":
    dim = 2048 + 5
    extractor = Hdf5FeatureExtractor("vinvl-r50c4-4setvg-rboxes-all-image", box_embedder=BasicBoxEmbedder(),
                                     all_image_box=True, all_image_prior=-10000)
  elif args.vmodel == "vinvl-r50c4-4setvg-rboxes-bk":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer(
      "vinvl", "R50C4_4setsvg", BasicBoxEmbedder(), args.vfreeze, train_transform="gpv1")
  elif args.vmodel == "vinvl-r50c4-4setvg-bk":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer("vinvl-r50c4-5setvg", "R50C4_4setsvg", BasicBoxEmbedder(), args.vfreeze)
  elif args.vmodel == "vinvl_backbone":
    dim = 2048 + 5
    extractor = VinvlBackboneImageFeaturizer("vinvl", "release", BasicBoxEmbedder(), args.vfreeze)
  elif args.vmodel in {"detr_boxes", "vinvl_boxes"}:
    dim = 2048 + 4*7
    extractor = FromPrecomputedBoxes(
      {"detr_boxes": "detr-coco-sce", "vinvl_boxes": "vinvl-boxes"}[args.vmodel],
      DetrBackbone(freeze=args.vfreeze),
      feature_extractor=BoxEmbedFeatureExtractor(
        box_coordinate_embed=NonLinearCoordinateEncoder([0.1, 0.05, 0.02]),
        post_rio=LayerNorm(),
      ),
      include_all_image_box=True,
      horizontal_flip=0.5,
      preload_bboxes=True,
    )
  else:
    raise NotImplementedError(args.vmodel)
  return extractor, dim