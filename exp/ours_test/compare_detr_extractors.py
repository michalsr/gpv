import torch

from exp.ours.data.dataset import GpvDataset, Task
from exp.ours.data.gpv_example import GPVExample
from exp.ours.image_featurizer.image_featurizer import ImageRegionFeatures, \
  PretrainedDetrRIOExtractor
from exp.ours.util import our_utils, image_utils, py_utils
from utils.detr_misc import nested_tensor_from_tensor_list


def main():
  py_utils.add_stdout_logger()

  boxes = PretrainedDetrRIOExtractor()
  boxes.eval()
  # boxes = PretrainedDetrFromBoxes("detr-eval")
  # boxes.eval()
  #
  detr = our_utils.get_detr_model()
  detr.eval()

  examples = GpvDataset(Task.CAPTIONING, "val").load()[:2]
  examples = [GPVExample(x.get_gpv_id(), Task.CAPTIONING, x.image_id, None) for x in examples]

  box_out: ImageRegionFeatures = boxes(**boxes.get_collate(False).collate(examples)[0])

  image_tensors = []
  for example in examples:
    trans = boxes.eval_transform
    img, size = image_utils.load_image_data(example, boxes.image_size)
    image_tensors.append(trans(img))
  images = nested_tensor_from_tensor_list(image_tensors)
  detr_out = detr(images)

  print(torch.max(torch.abs(box_out.boxes[0]-detr_out["pred_boxes"][0])))
  print(torch.max(torch.abs(box_out.features[0]-detr_out["detr_hs"][0, 0])))


if __name__ == '__main__':
  main()
