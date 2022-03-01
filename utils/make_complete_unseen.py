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

def new_train():
    train = []
    gpv_split = io.load_json_object('/data/michal5/gpv/learning_phase_data/coco_detection/original_split/train.json')
    for entry in gpv_split:
        if entry['category_name'] not in UNSEEN_COMBINED:
            train.append(entry)
    print(f'{len(train)} number of new train')
    
    if not os.path.exists('/data/michal5/gpv/learning_phase_data/coco_detection/seen_only'):
        os.mkdir('/data/michal5/gpv/learning_phase_data/coco_detection/seen_only')    
                     
    if not os.path.exists('/shared/rsaas/michal5/gpv_michal/coco_det_train_seen_only'):
        os.mkdir('/shared/rsaas/michal5/gpv_michal/coco_det_train_seen_only')
    io.dump_json_object(train,'/data/michal5/gpv/learning_phase_data/coco_detection/seen_only/train.json')
    io.dump_json_object(train,'/shared/rsaas/michal5/gpv_michal/coco_det_train_seen_only/train.json')  
def new_val():
    unseen_1_val = []
    unseen_1_test = []
    unseen_2_val = []
    unseen_2_test = []
    other_val = []
    other_test = []

    val_split = []
    test_split = []
    gpv_splits = [io.load_json_object('/data/michal5/gpv/learning_phase_data/'+'coco_detection/'+'original_split/'+'val.json'),io.load_json_object('/data/michal5/gpv/learning_phase_data/'+'coco_detection/'+'gpv_split/'+'test.json')]         
    for i,split in enumerate(gpv_splits):
      for entry in split:
            coco_classes_1 = []
            coco_classes_2 = []
            entry['query'] = f"Localize the {entry['category_name']}"
            if entry['category_name'] in UNSEEN1:
                
                if i ==0:
                    unseen_1_val.append(entry)
                else:
                    unseen_1_test.append(entry)
            if entry['category_name'] in UNSEEN2:
               
                if i ==0:
                    unseen_2_val.append(entry)
                else:
                    unseen_2_test.append(entry)
            else:
                if i ==0:
                    other_val.append(entry)
                else:
                    other_test.append(entry)
                    
                 
           
    print(f'{len(unseen_1_val)} is unseen 1 val length')
    print(f'{len(unseen_1_test)} is unseen 1 test length')
    print(f'{len(unseen_2_val)} is unseen 2 val length')
    print(f'{len(unseen_2_test)} is unseen 2 test length')
    print(f'{len(other_val)} is other val length')
    print(f'{len(other_test)} is other test length')
    if not os.path.exists('/shared/rsaas/michal5/gpv_michal/coco_det_val_all_single_phrase'):
        os.mkdir('/shared/rsaas/michal5/gpv_michal/coco_det_val_all_single_phrase')
                
    if not os.path.exists('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase'):
        os.mkdir('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase')    
                     
    if not os.path.exists('/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_1_single_phrase'):
        os.mkdir('/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_1_single_phrase')    
    io.dump_json_object(unseen_1_val,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase/val.json')
    io.dump_json_object(unseen_1_test,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_1_single_phrase/test.json')
    io.dump_json_object(unseen_1_val,'/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_1_single_phrase/val.json')
    io.dump_json_object(unseen_1_test,'/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_1_single_phrase/test.json')
    if not os.path.exists('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_2_single_phrase'):
        os.mkdir('/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_2_single_phrase')
    if not os.path.exists('/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_2_single_phrase'):
        os.mkdir('/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_2_single_phrase')
    io.dump_json_object(unseen_2_val,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_2_single_phrase/val.json')
    io.dump_json_object(unseen_2_test,'/data/michal5/gpv/learning_phase_data/coco_detection/unseen_group_2_single_phrase/test.json')
    io.dump_json_object(unseen_2_val,'/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_2_single_phrase/val.json')
    io.dump_json_object(unseen_2_test,'/shared/rsaas/michal5/gpv_michal/coco_det_val_all/unseen_group_2_single_phrase/test.json')
    

    if not os.path.exists('/data/michal5/gpv/learning_phase_data/coco_detection/seen_single_phrase'):
        os.mkdir('/data/michal5/gpv/learning_phase_data/coco_detection/seen_single_phrase')
    if not os.path.exists('/shared/rsaas/michal5/gpv_michal/coco_det_val_all/seen_single_phrase'):
        os.mkdir('/shared/rsaas/michal5/gpv_michal/coco_det_val_all/seen_single_phrase')
    io.dump_json_object(other_val,'/data/michal5/gpv/learning_phase_data/coco_detection/seen_single_phrase/val.json')
    io.dump_json_object(other_test,'/data/michal5/gpv/learning_phase_data/coco_detection/seen_single_phrase/test.json')
    io.dump_json_object(other_val,'/shared/rsaas/michal5/gpv_michal/coco_det_val_all/seen_single_phrase/val.json')
    io.dump_json_object(other_test,'/shared/rsaas/michal5/gpv_michal/coco_det_val_all/seen_single_phrase/test.json')
    




if __name__ == '__main__':
  new_train()