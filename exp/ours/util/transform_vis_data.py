import os 
import json 
import utils.io as io 
import torch 
import numpy as np
final_dict = {'image_ids':[],'pred_boxes':[],'queries':[],'rel':[]}

vis_data = io.load_json_object('/home/michal/gpv_michal/vis_data.json')
# print(vis_data[0]['image_ids'])
# for point in vis_data:
#     for i in range(len(point['image_ids'])):



#         final_dict['image_ids'].append(point['image_ids'][i])
#         print(point['image_ids'][i])
#         print(point['boxes'][i][0])
#         print(len(point['image_ids'][i]),len(point['boxes'][i]),len(point['queries'][i]))
#         rel = point['rel'][i]
#         pred_boxes = point['boxes'][i]
#         rel = torch.tensor(rel)
#         rel = rel.softmax(-1)[:,  0]
#         rel = rel.cpu().numpy()
#         ixs = np.argsort(rel)
#         print(ixs[0])
#         pred_boxes = pred_boxes[ixs[0]]
#         final_dict['pred_boxes'].append(pred_boxes)
#         final_dict['queries'].append(point['queries'][i])
#         final_dict['rel'].append(rel[0])



for entry in vis_data:
    for point in entry['ind_mapping']:
        image_id,_,rel,pred_boxes = entry['ind_mapping'][point]
        rel = torch.tensor(rel)
      
        rel = rel.softmax(-1)[:,  0]

        rel = rel.cpu().numpy()
        ixs = np.argsort(rel)
        print(rel[ixs[0]],rel[ixs[-1]],'rel')
        pred_boxes = pred_boxes[ixs[-1]]
        final_dict['image_ids'].append(image_id)
        final_dict['pred_boxes'].append(pred_boxes)
        query = entry['queries'][int(point)]
        final_dict['queries'].append(query)
        final_dict['rel'].append(rel[ixs[-1]])
io.dump_json_object(final_dict,'/home/michal/gpv_michal/outputs/syn_vis_data/html_files.json')