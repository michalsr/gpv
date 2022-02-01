from decimal import Subnormal
import utils.io as io
import json 
import numpy as np
from tqdm import tqdm 
UNSEEN_COMBINED = ['bed', 'bench', 'book', 'cell phone', 'horse', 
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']



web_cat_to_image_id = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_to_image_id.json')
all_web_concepts = list(web_cat_to_image_id.keys())
coco_cat_to_web = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/coco_cat_to_web_cat.json')

#no contrastive part 
#for each image in coco category fill in prompt 
#when loading sample 
def determine_if_contains_unseen(entry):
    coco_classes = []
    if len(entry['coco_categories']['seen']) != 0:
        for c in entry['coco_categories']['seen']:
            coco_classes.append(c)
        if len(entry['coco_categories']) != 0:
            for c in entry['coco_categories']['unseen']:
                coco_classes.append(c)
    if len(coco_classes) != 0:
        if all(c in UNSEEN_COMBINED for c in coco_classes):
            if entry['verb'] != None and entry['verb'] != 'null':
                return True 
    return False
# def get_list_of_queries():
#     subs= {
#   "DT_OBJ": [
#     "this object", "this entity", "this thing",
#     "the object", "the entity",
#     "that object", "that entity",  "that thing"
#   ],
#   "DT": ["the", "this", "that"],
#   "OBJ": ['object', 'entity'],
#   "CMD": ["Describe", "State", "Specify", "Name"],
#   "NAME": ["Describe", "Specify", "Name", "Classify"],
#   "CAP": ["Describe", "Caption", "Generate a caption for"],
#   "WH": ["What", "Which"]
# }
#     verb_questions  = [
#   "What is DT_OBJ doing?",
#   "What action is DT_OBJ taking?",
#   "What action is DT_OBJ performing?",
#   "What action is DT_OBJ carrying out?",
#   "What action is DT_OBJ doing?",
#   "What activity is DT_OBJ doing?",
#   "CMD the action being taken by DT_OBJ.",
#   "CMD the activity DT_OBJ is doing.",
#   "CMD what DT_OBJ is doing.",
# ]
#     final_queries = []
#     for q in verb_questions:
#         if 'DT_OBJ' in n:
#             for sub in subs['DT_OBJ']:
#                 print(sub)

#                 final_queries.append(n.replace('DT_OBJ',sub))
#         elif 'OBJ' in n:
#             for sub in subs['OBJ']:
#                 final_queries.append(n.replace('OBJ',sub))
#     return final_queries

def main():
    subs= {
  "DT_OBJ": [
    "this object", "this entity", "this thing",
    "the object", "the entity",
    "that object", "that entity",  "that thing"
  ],
  "DT": ["the", "this", "that"],
  "OBJ": ['object', 'entity'],
  "CMD": ["Describe", "State", "Specify", "Name"],
  "NAME": ["Describe", "Specify", "Name", "Classify"],
  "CAP": ["Describe", "Caption", "Generate a caption for"],
  "WH": ["What", "Which"]
}
    verb_questions  = ["What is DT_OBJ doing?","What action is DT_OBJ taking?","What action is DT_OBJ performing?","What action is DT_OBJ carrying out?", "What action is DT_OBJ doing?","What activity is DT_OBJ doing?","CMD the action being taken by DT_OBJ.","CMD the activity DT_OBJ is doing.","CMD what DT_OBJ is doing.",]
    web_training_info = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/train_image_info.json')
    action_with_obj_data = []
    id_to_use = 0



    for entry in web_training_info:
        if determine_if_contains_unseen(entry):
            question_list = []
            for question in verb_questions:
                        if 'DT_OBJ' in question:
                            for sub in subs['DT_OBJ']:
                                question = question.replace('DT_OBJ',sub)
                        if 'CMD' in question:
                            for sub in subs['CMD']:
                                question = question.replace('CMD',sub)
                                question_list.append(question)
            if len(question_list)>0:
                for query in question_list:
                    if type(entry['verb']) == list:
                        for v in entry['verb']:
                            ex = {}
                            ex['image'] = {'image_id':entry['image']['image_id']}
                            ex['query'] = query
                            ex['answer'] = v
                            ex['gpv_id'] = f"vqa-action-with-object-{str(entry['noun'])}-{str(entry['verb'])}-{str(entry['image']['image_id'])}-{str(id_to_use)}"
                            id_to_use+= 1
                            action_with_obj_data.append(ex)
                    else:
                        ex = {}
                        ex['image'] = {'image_id':entry['image']['image_id']}
                        ex['query'] = query
                        ex['answer'] = entry['verb']
                        ex['gpv_id'] = f"vqa-action-with-object-{str(entry['noun'])}-{str(entry['verb'])}-{str(entry['image']['image_id'])}-{str(id_to_use)}"
                        id_to_use+= 1
                        action_with_obj_data.append(ex)



    print(len(action_with_obj_data))
    io.dump_json_object(action_with_obj_data,'/data/michal5/gpv/lessons/vqa_action_no_obj.json')
    io.dump_json_object(action_with_obj_data,'/shared/rsaas/michal5/gpv_michal/lessons/vqa_action_no_obj.json')
    




if __name__ == '__main__':
    # make_coco_cat_to_web_cat()
    # coco_cat_to_web = io.load_json_object('/data/michal5/web_training_info/coco_cat_to_web_cat_seen.json')

    main()




