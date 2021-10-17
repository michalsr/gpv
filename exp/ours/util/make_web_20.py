import json 
import utils.io as io
UNSEEN1 = ['bed', 'bench', 'book', 'cell phone', 'horse', 'tv',
             'sheep', 'suitcase', 'surfboard', 'wine glass']
UNSEEN2 = ['banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'remote']


# web_20_entries = []
# web_80 = io.load_json_object('/data/michal5/train_image_info.json')

# for entry in web_80:
#     print(entry.keys())
#     classes = set()
#     coco_categories = entry['coco_categories']
#     if len(coco_categories['seen']) != 0:
#         for c in coco_categories['seen']:
#             classes.add(c)
#     if len(coco_categories['unseen']) != 0:
#         for c in coco_categories['unseen']:
#             classes.add(c)
#     if all(c in UNSEEN1 for c in classes):
#         web_20_entries.append(entry)
# print(len(web_20_entries))
# print(len(web_80))
# print(len(web_20_entries))
# io.dump_json_object(web_20_entries,'/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')
tasks = ['coco_classification','coco_detection','vqa','coco_captions']
ind = ['train','val','test']
for t in tasks:
    for i in ind:
        new_test = []
        previous_test = io.load_json_object(f'/data/michal5/gpv/learning_phase_data/{t}/held_out_test/{i}.json')
     
        
        for entry in previous_test:
            classes = set()
            coco_categories = entry['coco_categories']
            
            if len(coco_categories['seen']) != 0:
                for c in coco_categories['seen']:
                    classes.add(c)
            if len(coco_categories['unseen']) != 0:
                for c in coco_categories['unseen']:
                    classes.add(c)
       
            if all(c in UNSEEN1 for c in classes):
                new_test.append(entry)
        print(len(new_test),len(previous_test))
        assert len(new_test) < len(previous_test)
        io.dump_json_object(new_test,f'/data/michal5/gpv/learning_phase_data/{t}/held_out_10/{i}.json')


