import json 
import utils.io as io

new_train_data = []
old_train_data = io.load_json_object('/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')
prefixes = ["q", "1n", "1v", "1a", "2a", "2v"]
for i,entry in enumerate(old_train_data):
    new_entry = entry
    new_entry['web_id'] = {}
    for p in prefixes:
        new_entry['web_id'][p] = f"{p}_{i}"
    new_train_data.append(new_entry)
io.dump_json_object(new_train_data,'/data/michal5/gpv/learning_phase_data/web_20/test_image_info.json')