import logging
from os.path import join, dirname

DATA_DIR = "/Users/chris/data/"

FASTER_RCNN_SOURCE = join(DATA_DIR, "faster-rcnn", "trainval_resnet101_faster_rcnn_genome_36.tsv")
VINVL_SOURCE = join(DATA_DIR, "vinvl")
VINVL_COCO_FEATURES = join(VINVL_SOURCE, "model_0060000")
VINVL_MODEL_WEIGHTS = join(VINVL_SOURCE, "models", "vinvl_vg_x152c4.pth")

COCO_SOURCE = join(DATA_DIR, "coco")
COCO_ANNOTATIONS = join(COCO_SOURCE, "annotations")
VQA2_SOURCE = join(DATA_DIR, "vqa-2.0")

TORCHVISION_CACHE_DIR = join(DATA_DIR, "torchvision-cache")

CACHE_DIR = join(dirname(dirname(dirname(__file__))), "data-cache")
PRECOMPUTED_FEATURES_DIR = join(CACHE_DIR, "precomputed-features")

GPV_DIR = join(DATA_DIR, "gpv")
COCO_IMAGES = join(GPV_DIR, "learning_phase_data/coco/images")

PRETRAINED_DETR_MODELS = {
 "coco_sce": join(GPV_DIR, "detr", "detr_coco_sce.pth"),
 "coco": join(GPV_DIR, "detr", "detr_coco.pth")
}

SOURCE_DIR = join(GPV_DIR, "learning_phase_data")
GPV1_VOC = join(SOURCE_DIR, "vocab/vocab.json")
GPV1_VOC_EMBED = join(SOURCE_DIR, "vocab/vocab_embed.npy")

CLASSIFICATION = join(SOURCE_DIR, "coco_classification")
IMAGE_DIR = join(SOURCE_DIR, "coco/images")

VISUALIZATION_DIR = "/Users/chris/Desktop/gpv-visualize"

BEST_STATE_NAME = "best-state.pth"
