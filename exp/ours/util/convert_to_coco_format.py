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
unseen_gpv_annotations = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase/val.json')
coco_annotations = {'images':[],'annotations':[],'categories':[]}
images_processed = []
box_ids = 0
for example in unseen_gpv_annotations:
    image_id = str(example['image']['image_id'])
    subset = example['image']['subset']
    final_id = image_id.zfill(12)
    file_name = f'/data/michal5/gpv/learning_phase_data/coco/images/{subset}/COCO_{subset}_{final_id}.jpg'
   
    img = cv2.imread(file_name)

    img_h,img_w,c = img.shape
    category_id = example['category_id']
    if image_id not in images_processed:
        image_dict = {"file_name":file_name,'height':img_h,'width':img_w,'category_id':category_id,'id':example['id']}
        coco_annotations['images'].append(image_dict)
        images_processed.append(image_id)
    for i,box in enumerate(example['boxes']):
        x,y,w,h = box 
        annotation_dict = {'bbox':[x,y,w,h],'category_id':category_id,'image_id':image_id,'id':example['instance_ids'][i],'iscrowd':0,'area':w*h}
        coco_annotations['annotations'].append(annotation_dict)
for cat in COCO_CATEGORIES_TO_ID:
    cat_dict = {'id':COCO_CATEGORIES_TO_ID[cat],'name':cat}
    coco_annotations['categories'].append(cat_dict)
io.dump_json_object(coco_annotations,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase_coco/val.json')




