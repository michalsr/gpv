import json 
import utils.io as io
import os

#validation 
UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
def new_val():
    val_split = []
    test_split = []
    gpv_splits = [io.load_json_object('/data/michal5/gpv/learning_phase_data/'+'vqa/'+'gpv_split/'+'val.json'),io.load_json_object('/data/michal5/gpv/learning_phase_data/'+'vqa/'+'gpv_split/'+'test.json')]         
    for split in gpv_splits:
      for entry in split:
            coco_classes = []
            if len(entry['coco_categories']['seen']) != 0:
                for c in entry['coco_categories']['seen']:
                    coco_classes.append(c)
            if len(entry['coco_categories']) != 0:
                for c in entry['coco_categories']['unseen']:
                    coco_classes.append(c)
            if len(coco_classes) != 0:
                if all(c in UNSEEN1 for c in coco_classes):
                    val_split.append(entry)
                if all(c in UNSEEN2 for c in coco_classes):
                    test_split.append(entry)
            # if not  all(c in unseen_categories.values() for c in coco_classes) and int(entry_id) not in ids_used:
            #     new_split.append(entry)
            #     ids_used.add(int(entry_id))
    io.dump_json_object(val_split,'/data/michal5/gpv/learning_phase_data/vqa/unseen_10/val.json')
    io.dump_json_object(val_split,'/shared/rsaas/michal5/gpv_michal/vqa_unseen/val.json')
    io.dump_json_object(test_split,'/data/michal5/gpv/learning_phase_data/vqa/unseen_10/test.json')
    io.dump_json_object(test_split,'/shared/rsaas/michal5/gpv_michal/vqa_unseen/test.json')


if __name__ == '__main__':
  new_val()