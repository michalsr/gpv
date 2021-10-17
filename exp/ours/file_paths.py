from os.path import join, dirname

DATA_DIR = "/data/michal5/"

FASTER_RCNN_SOURCE = join(DATA_DIR, "faster-rcnn", "trainval_resnet101_faster_rcnn_genome_36.tsv")
VINVL_SOURCE = join(DATA_DIR, "vinvl")
VINVL_COCO_FEATURES = join(VINVL_SOURCE, "model_0060000")
VINVL_MODEL_WEIGHTS = join(VINVL_SOURCE, "models", "vinvl_vg_x152c4.pth")

NOCAPS_VAL_IMAGE_INFO = join(DATA_DIR, "nocaps", "nocaps_val_image_info.json")
COCO_SOURCE = join(DATA_DIR, "coco")
COCO_ANNOTATIONS = join(COCO_SOURCE, "annotations")
VQA2_SOURCE = join(DATA_DIR, "vqa-2.0")

TORCHVISION_CACHE_DIR = join(DATA_DIR, "torchvision-cache")

WEB_IMAGES_DIR = "/data/michal5/gpv/learning_phase_data/web_data/images"

#CACHE_DIR = join(dirname(dirname(dirname(__file__))), "data-cache")
# CACHE_DIR = "/home/amitak/gpv-2/gpv/data-cache"
CACHE_DIR = "/data/michal5/gpv/data-cache/precomputed-features"
PRECOMPUTED_FEATURES_DIR = "/data/michal5/gpv/data-cache/precomputed-features/"
WEBQA_ALL_FIFTH_ANSWERS = join(CACHE_DIR, "webqa_fifth_answers.json")

GPV_DIR = join(DATA_DIR, "gpv")
WEBQA_DIR = join(GPV_DIR, "learning_phase_data/web_20")
COCO_IMAGES = join(GPV_DIR, "learning_phase_data/coco/images")

PRETRAINED_DETR_MODELS = {
 "coco_sce": join(GPV_DIR, "detr", "detr_coco_sce.pth"),
 "coco": join(GPV_DIR, "detr", "detr_coco.pth")
}

OPENSCE_HOME = join(DATA_DIR, "git/opensce")
OPENSCE_SAMPLES = join(OPENSCE_HOME, "samples_w_prompts")
OPENSCE_IMAGES = join(OPENSCE_HOME, "images")
OPENSCE_SYN = join(OPENSCE_HOME, "opensce_synonyms.json")
OPENSCE_CATS = join(OPENSCE_HOME, "opensce_categories.csv")

SOURCE_DIR = join(GPV_DIR, "learning_phase_data")
WEBQA80_ANSWERS = join(SOURCE_DIR, "vocab/web_80_answers.json")
GPV1_VOC = join(SOURCE_DIR, "vocab/vocab.json")
GPV1_VOC_EMBED = join(SOURCE_DIR, "vocab/vocab_embed.npy")

CLASSIFICATION = join(SOURCE_DIR, "coco_classification")

VISUALIZATION_DIR = "/home/amitak/gpv-2/gpv/gpv-visualize"

BEST_STATE_NAME = "best-state.pth"

