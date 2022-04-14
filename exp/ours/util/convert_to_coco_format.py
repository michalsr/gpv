import json 
import os 
import utils.io as io 
import cv2

def get_coco_categories():
  coco_file = '/shared/rsaas/michal5/gpv_michal/exp/ours/data/coco_categories.json'
  return io.load_json_object(coco_file)
COCO_ID_TO_CATEGORY = {x["id"]: x["name"] for x in get_coco_categories()}
COCO_CATEGORIES = list(COCO_ID_TO_CATEGORY.values())
COCO_CATEGORIES_TO_ID = {k: i for i, k in enumerate(COCO_CATEGORIES)}
unseen_gpv_annotations = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase/test.json')
coco_annotations = {'images':[],'annotations':[],'categories':[]}
images_processed = []
categories_listed = set()
box_ids = 0
for example in unseen_gpv_annotations:
    image_id = str(example['image']['image_id'])
    subset = example['image']['subset']
    final_id = image_id.zfill(12)
    file_name = f'/data/michal5/gpv/learning_phase_data/coco/images/{subset}/COCO_{subset}_{final_id}.jpg'
    
    img = cv2.imread(file_name)


    img_h,img_w,c = img.shape
    category_id = example['category_id']
    categories_listed.add(example['category_id'])
    if image_id not in images_processed:
        image_dict = {"file_name":file_name,'height':img_h,'width':img_w,'category_id':category_id,'id':float(image_id),'image_id':float(image_id)}
        coco_annotations['images'].append(image_dict)
        images_processed.append(image_id)
    normalized_gt = all(all(val <= 1.0 for val in b) for b in example['boxes'])
    #print(example['boxes'])
    # if not normalized_gt:
    #     # convert to relative coordinates
    #     # TODO its a bit of hack to check this by looking coordinates > 1.0
    #     # but we need this check atm since OpenSCE stores relative scaling
    #     # coco uses absolute
    #     H,W,C = img.shape
    #     example['boxes'][:, 0] = example['boxes'][:, 0] / W
    #     example['boxes'][:, 1] = example['boxes'][:, 1] / H
    #     example['boxes'][:, 2] = example['boxes'][:, 2] / W
    #     example['boxes'][:, 3] = example['boxes'][:, 3] / H
    for i,box in enumerate(example['boxes']):
        H,W,C = img.shape
        x,y,w,h = box
        if not normalized_gt:
            x = x/W 
            y = y/H
            w = w/W
            h = h/H

        
        annotation_dict = {'bbox':[x,y,w,h],'category_id':category_id,'image_id':float(image_id),'id':float(example['instance_ids'][i]),'iscrowd':0,'area':w*h}
        coco_annotations['annotations'].append(annotation_dict)
for cat in categories_listed:
    cat_dict = {'id':cat,'name':COCO_ID_TO_CATEGORY[cat]}
    coco_annotations['categories'].append(cat_dict)
io.dump_json_object(coco_annotations,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase_coco/test.json')




