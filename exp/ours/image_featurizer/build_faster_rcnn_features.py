import base64
import csv
import json
from os.path import join

import h5py
from exp.ours import config
import numpy as np


def main():
  file = h5py.File(join(config.CACHE_DIR, "trainval-mscoco-faster-rcnn.hdf5"), "w")
  source_file = config.FASTER_RCNN_SOURCE

  with open(source_file, "r+b") as source_f:
    for line in source_f:
      parts = line.split(b"\t")
      image_id, image_w, image_h, num_boxes = [int(x) for x in parts[:4]]
      features = np.frombuffer(
        base64.decodebytes(parts[5]),
        dtype=np.float32).reshape((num_boxes, -1))
      bboxes = np.frombuffer(
        base64.decodebytes(parts[4]),
        dtype=np.float32).reshape((num_boxes, -1))
      image_group = file.create_group(str(image_id))
      image_group.create_dataset("bboxes", data=bboxes)
      image_group.create_dataset("features", data=features)
      image_group.create_dataset("image_size", data=np.array([
        int(image_w), int(image_h)
      ], dtype=np.int32))


if __name__ == '__main__':
  main()