import logging
import os
import struct
from collections import defaultdict
from multiprocessing import Lock, Pool
from os import listdir, mkdir
from os.path import join, isdir, exists, relpath
from typing import Tuple, Optional, Union

import imagesize
import numpy as np
import skimage.io as skio
from PIL import Image
from skimage.transform import resize

from exp.ours import file_paths
from exp.ours.util import py_utils

DUMMY_IMAGE_ID = object()


def imageid_from_file(filename):
  return int(filename.split("_")[-1].split(".")[0])


def file_from_imageid(image_subset: str, image_id: int):
  return f'COCO_{image_subset}_{str(image_id).zfill(12)}.jpg'


def _build_image_file_map():
  file_map = {}
  for dirname in listdir(file_paths.COCO_IMAGES):
    dirname = join(file_paths.COCO_IMAGES, dirname)
    if not isdir(dirname):
      continue
    for file in listdir(dirname):
      file_map[imageid_from_file(file)] = join(dirname, file)

  return file_map


_IMAGE_ID_TO_FILE_MAP = None


def get_image_file(image_id):
  if isinstance(image_id, dict):
    # Legacy image_id were dictionaries
    filename = file_from_imageid(image_id["subset"], image_id["image_id"])
    image_file = join(file_paths.COCO_IMAGES, image_id["subset"], filename)
    if not exists(image_file):
      raise ValueError(f"Missing file {image_file} for image {image_id}")
    return image_file

  if isinstance(image_id, str):
    if image_id.startswith("imsitu/"):
      return join(file_paths.IMSITU_IMAGE_DIR, image_id[len("imsitu/"):])

    # I didn't use sensible prefixes for opensce and web images, so as a hack we check
    # for their correct source based on the fact the web images do not have
    # a file prefix.
    if image_id.endswith(".jpg"):
      return join(file_paths.OPENSCE_IMAGES, image_id)
    else:
      return join(file_paths.WEB_IMAGES_DIR, image_id)

  # Legacy coc image_ids were integer, look up its filepath here
  global _IMAGE_ID_TO_FILE_MAP
  if _IMAGE_ID_TO_FILE_MAP is None:
    _IMAGE_ID_TO_FILE_MAP = _build_image_file_map()

  if image_id not in _IMAGE_ID_TO_FILE_MAP:
    raise ValueError(f"Missing file for image {image_id} in {file_paths.COCO_IMAGES}")
  return _IMAGE_ID_TO_FILE_MAP[image_id]


def get_coco_int_id(image_id) -> int:
  if isinstance(image_id, int):
    return image_id
  return int(image_id.split("-")[-1])


_IMAGE_ID_TO_SIZE_MAP = {}


def get_image_size(image_id):
  if image_id in _IMAGE_ID_TO_SIZE_MAP:
    return _IMAGE_ID_TO_SIZE_MAP[image_id]

  img_file = get_image_file(image_id)
  size = imagesize.get(img_file)

  _IMAGE_ID_TO_SIZE_MAP[image_id] = size
  return size


def crop_img(img: Union[np.ndarray, Image.Image], crop):
  if crop is None:
    return img
  x, y, w, h = crop

  if isinstance(img, np.ndarray):
    H, W = img.shape[:2]
  else:
    W, H = img.size

  if all(c <= 1.0 for c in crop):
    # Assume the crop is in normalized coordinates
    x, y, w, h = x*W, y*H, w*W, h*H

  if w < 5: w = 5
  if h < 5: h = 5
  x1 = x - 0.2 * w
  x2 = x + 1.2 * w
  y1 = y - 0.2 * h
  y2 = y + 1.2 * h
  x1, x2 = [min(max(0, int(z)), W) for z in [x1, x2]]
  y1, y2 = [min(max(0, int(z)), H) for z in [y1, y2]]
  if isinstance(img, np.ndarray):
    return img[y1:y2, x1:x2]
  else:
    return img.crop((x1, y1, x2, y2))


def get_box_key(box):
  crop = np.array(box, dtype=np.float32).tobytes()
  return py_utils.consistent_hash(crop)


def get_cropped_img_key(image_id, crop=None):
  image_id = str(image_id)

  # Replace "/" to avoid hdf5 building hierarchical groups
  if "-" in image_id:
    raise ValueError()
  image_id = str(image_id).replace("/", "-")
  if crop is None:
    return str(image_id)
  return f"{image_id}-{get_box_key(crop)}"


def load_image_pil(image_id, crop) -> Image.Image:
  img_file = get_image_file(image_id)
  img = Image.open(img_file).convert("RGB")
  if crop:
    img = crop_img(img, crop)
  return img


def load_image_data(example, size):
  if hasattr(example, "image_file"):
    # Allow examples to specify their own file
    return load_image_ndarray(example.image_file, size, example.crop)
  return load_image_ndarray(get_image_file(example.image_id), size, example.crop)


def load_image_ndarray(img_file, image_size=None, crop=None) -> Tuple[np.ndarray, Tuple[int, int]]:
  try:
    img = skio.imread(img_file)
    if len(img.shape) == 2:
      img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
    else:
      img = img[:, :, :3]
  except OSError as e:
    raise ValueError(f"Error reading image {img_file}")

  if crop:
    img = crop_img(img, crop)

  original_image_size = img.shape[:2]  # HxW

  if image_size:
    img = resize(img, image_size, anti_aliasing=True)
    img = (255 * img).astype(np.uint8)
  return img, original_image_size


def get_hdf5_image_file(dataset_name, features_name):
  if not exists(file_paths.PRECOMPUTED_FEATURES_DIR):
    mkdir(file_paths.PRECOMPUTED_FEATURES_DIR)
  return join(file_paths.PRECOMPUTED_FEATURES_DIR, dataset_name, features_name + ".hdf5")


def get_hdf5_files(feature_name):
  files = []
  for dataset_dir in listdir(file_paths.PRECOMPUTED_FEATURES_DIR):
    f = join(file_paths.PRECOMPUTED_FEATURES_DIR, dataset_dir, feature_name+".hdf5")
    if exists(f):
      files.append(f)
  return files
