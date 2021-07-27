from os.path import join

from exp.ours.data.dataset import GpvDataset
from exp.ours.data.source_data import load_instances
from utils.io import load_json_object


class CocoCroppedDataset:

  def __init__(self):
    for part in ["train", "val", "test"]:
      target_file = join("", "coco_classification", "split_txt", f"{part}.json")
      print(f"Loading instances from {target_file}")
      data = load_json_object(target_file)
