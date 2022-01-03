import os 
import magic 
import cv2
import numpy as np
import networkx as nx 
import json 
import pickle
import yaml
import numpy as np
import gzip
import scipy.io

UNSEEN = ['bed', 'bench', 'book', 'cell phone', 'horse', 'remote',
             'sheep', 'suitcase', 'surfboard', 'wine glass','banana', 'baseball bat', 'bottle', 'broccoli', 'donut',
             'hot dog', 'keyboard', 'laptop', 'train', 'tv']
# def get_random_img_of_concept(concept):
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
def create_grid(list_of_imgs):
    new_im_list = []
    im_1_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{list_of_imgs[0]}')
    im_1 = cv2.resize(im_1_1,(64,64))
    im_2_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{list_of_imgs[1]}')
    im_2 = cv2.resize(im_2_1,(64,64))
    im_3_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{list_of_imgs[2]}')
    im_3 = cv2.resize(im_3_1,(64,64))
    im_4_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{list_of_imgs[3]}')
    im_4 = cv2.resize(im_4_1,(64,64))
    im_grd = concat_tile([[im_1, im_2],
                       [im_3,im_4]])
    return im_grd 
def get_random_imgs(concept,category_to_web_id,net,json_dict):
    list_of_images = list(category_to_web_id.keys())
    unseen = UNSEEN.copy()
    unseen.remove(concept)
    unseen.remove('remote')
  
    c = np.random.choice(unseen,3)
    img_set = []
    for neighbor_c in c.tolist():

        entry = json_dict['cat_to_num'][neighbor_c]
        edges = net[entry]
        chosen_edge = np.random.choice(list(edges.keys()),1)
        chosen_edge_name = net.nodes[chosen_edge.tolist()[0]]['title']
        potential_imgs = json_dict['category_to_image_id'][chosen_edge_name]
        chosen_img = np.random.choice(potential_imgs,1)
        img_set.append(chosen_img[0])
    return img_set

def dump_json_object(dump_object, file_name, compress=False, indent=4):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode('utf8')))
    else:
        write(file_name, data, 'w')
def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)

def load_json_object(file_name, compress=False):
    if compress:
        return json.loads(gzip.decompress(read(file_name)).decode('utf8'))
    else:
        return json.loads(read(file_name, 'r'))
def read(file_name, mode='rb'):
    with open(file_name, mode) as f:
        return f.read()
def make_reverse_image_id_to_category(category_to_web_id):
    new_dict = {}
    for k in category_to_web_id.keys():
        for img in category_to_web_id[k]:
            new_dict[img] = k 
    return new_dict
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
def load_json_information():
    web_qa_id_to_value = load_json_object('/shared/rsaas/michal5/gpv_michal/query_to_web_id.json')
    category_names = [k for k in web_qa_id_to_value.keys()]
    category_to_coco_id = load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_coco_id.json')
    category_to_image_id = load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_to_image_id.json')
    category_to_pos = load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/category_to_pos.json')
    dict_list = {'category_names':category_names,'category_to_coco_id':category_to_coco_id,'category_to_image_id':category_to_image_id,
    'category_to_pos':category_to_pos,'web_id_to_value':web_qa_id_to_value}
    return dict_list 
dict_list = load_json_information()
dict_list['cat_to_num'] = load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/cat_to_num_id.json')
dict_list['image_id_to_category'] = make_reverse_image_id_to_category(dict_list['category_to_image_id'])
net = nx.read_gpickle("/shared/rsaas/michal5/gpv_michal/word_graph.gpickle")
get_random_imgs('bed',dict_list['category_to_image_id'],net,dict_list)
entry_to_grid = {}
noun_neighbors = []
adj_dict = {}
for c in UNSEEN:
    if c != 'remote':
#         if not (os.path.exists('/shared/rsaas/michal5/gpv_michal/grid_concepts/'+unseen_c+'/')):
#             os.mkdir(f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/')
#         print(f'Making grids for concept {unseen_c}')
        noun_neighbors = []
        entry = dict_list['cat_to_num'][unseen_c]
        edges = net[entry]
        for e in edges:
            if net.nodes[e]['pos']['adj'] != None:
                noun_neighbors.append(e)
        for entry in noun_neighbors:
            if net.nodes[entry]['pos']['adj'] not in adj_dict:
                adj_dict[net.nodes[entry]['pos']['adj']] = set()
        adj_dict[net.nodes[entry]['pos']['adj']].add(net.nodes[entry]['title'])
        neighbor_entry = net[entry]

   
        for neighbor_edge in neighbor_entry:
            if net.nodes[neighbor_edge]['pos']['adj'] != None:
                if net.nodes[neighbor_edge]['pos']['adj'] not in adj_dict:
                    adj_dict[net.nodes[neighbor_edge]['pos']['adj']] = set()
                adj_dict[net.nodes[neighbor_edge]['pos']['adj']].add(net.nodes[neighbor_edge]['title'])
print(adj_dict.keys())

#             i = 0
        
               

# for e in edges:
#     if net.nodes[e]['pos']['adj'] != None:
#         noun_neighbors.append(e)
# for entry in noun_neighbors:
#     if net.nodes[entry]['pos']['adj'] not in adj_dict:
#         adj_dict[net.nodes[entry]['pos']['adj']] = set()
#     adj_dict[net.nodes[entry]['pos']['adj']].add(net.nodes[entry]['title'])
#     neighbor_entry = net[entry]

   
#     for neighbor_edge in neighbor_entry:
#         if net.nodes[neighbor_edge]['pos']['adj'] != None:
#             if net.nodes[neighbor_edge]['pos']['adj'] not in adj_dict:
#                 adj_dict[net.nodes[neighbor_edge]['pos']['adj']] = set()
#             adj_dict[net.nodes[neighbor_edge]['pos']['adj']].add(net.nodes[neighbor_edge]['title'])

# blue = adj_dict['yellow']
# print(np.random.choice(list(blue),4))
# blue_categories = ['yellow cap','yellow food','yellow easter egg','yellow hearts']
# dress_list_of_imgs = dict_list['category_to_image_id'][blue_categories[0]]
# eye_list_of_imgs = dict_list['category_to_image_id'][blue_categories[1]]
# toy_list_of_imgs = dict_list['category_to_image_id'][blue_categories[2]]
# jacket_list_of_imgs = dict_list['category_to_image_id'][blue_categories[3]]
# img_1_pic = np.random.choice(list(dress_list_of_imgs),1)
# img_2_pic = np.random.choice(list(eye_list_of_imgs),1)
# img_3_pic = np.random.choice(list(toy_list_of_imgs),1)
# img_4_pic = np.random.choice(list(jacket_list_of_imgs),1)
# img_1_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img_1_pic[0]}')
# img_1 = cv2.resize(img_1_1,(224,224))
# img_2_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img_2_pic[0]}')
# img_2 = cv2.resize(img_2_1,(224,224))
# img_3_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img_3_pic[0]}')
# img_3 = cv2.resize(img_3_1,(224,224))
# img_4_1 = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img_4_pic[0]}')
# img_4 = cv2.resize(img_4_1,(224,224))
# im_tile = concat_tile([[img_1, img_2],
#                        [img_3,img_4]])
# cv2.imwrite('/shared/rsaas/michal5/gpv_michal/adj_grid.jpg',im_tile)


# for unseen_c in UNSEEN:
#     if unseen_c != 'remote':
#         if not (os.path.exists('/shared/rsaas/michal5/gpv_michal/grid_concepts/'+unseen_c+'/')):
#             os.mkdir(f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/')
#         print(f'Making grids for concept {unseen_c}')
#         entry = dict_list['cat_to_num'][unseen_c]
#         edges = net[entry]

#         for e in edges:
#             i = 0
#             if net[entry][e]['edge'] == 'red':
#                 entry_title = net.nodes[e]['title']
#                 print(f'Making grid for imgs in {entry_title}')
#                 if not os.path.exists(f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/{entry_title}'):
#                     os.mkdir(f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/{entry_title}')
#                 if entry_title not in entry_to_grid:
#                     entry_to_grid[entry_title] = {}
            
#                 list_of_images = dict_list['category_to_image_id'][entry_title]
#                 for j,im in enumerate(list_of_images):
#                     if im not in entry_to_grid[entry_title].keys():
#                         entry_to_grid[entry_title][im] = {}
#                     random_imgs = get_random_imgs(unseen_c,dict_list['category_to_image_id'],net,dict_list)
#                     grid_imgs = [random_imgs[0],random_imgs[1],random_imgs[2]]
#                     index = np.random.choice(4,1)
#                     grid_imgs.insert(index[0],im)
#                     entry_to_grid[entry_title][im]['id'] = index[0]
#                     grid = create_grid(grid_imgs)
#                     entry_to_grid[entry_title][im]['title'] = f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/{entry_title}/img_{i}.jpg'
#                     img_title = f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/{entry_title}/img_{i}.jpg'
#                     entry_to_grid[entry_title][im]['im_d'] = f'/shared/rsaas/michal5/gpv_michal/grid_concepts/{unseen_c}/{entry_title}/img_{i}.jpg'
#                     cv2.imwrite(img_title,grid)
#                     i+=1
#                     print(f'Made grid for {j} out of {len(list_of_images)}')
# dump_json_object(entry_to_grid,'/shared/rsaas/michal5/gpv_michal/grid_concepts.json')



# img_1_1 = cv2.imread('/data/michal5/gpv/learning_phase_data/web_data/images/5b41fff59b79df3ca58aafd047dac5e29228c96861721bd3f54b07e2c7c84ff5')
# img_1 = cv2.resize(img_1_1,(224,224))
# img_2_1 = cv2.imread('/data/michal5/gpv/learning_phase_data/web_data/images/5b41d736eae336f2eb2b036c9627ed8344aa94886a3d47af634edcaef387517a')
# img_2 = cv2.resize(img_2_1,(224,224))
# img_3_1 = cv2.imread('/data/michal5/gpv/learning_phase_data/web_data/images/5b41b49b9f6c5e50fd45bd5b6bd20c5bb03be3488c727a6d881c6bb06e62a866')
# img_3 = cv2.resize(img_3_1,(224,224))
# img_4_1 = cv2.imread('/data/michal5/gpv/learning_phase_data/web_data/images/5b410916ad019d876e949edb747dc15a743c76aa9d4b6b9f34116b2db74e361f')
# img_4 = cv2.resize(img_4_1,(224,224))
# im_tile = concat_tile([[img_1, img_2],
#                        [img_3,img_4]])
# cv2.imwrite('/shared/rsaas/michal5/gpv_michal/test_grid.jpg',im_tile)

# def pick_concept(concepts,number)

# #map coco concepts to image ids 
# for i in range(concepts):
#     for j in range(num_grids):
        #find image
        #find three other images from other concepts


 #find  web concepts that share an adjective
#for all coco concepts
#get web concept thtat share noun
#for each image in web concept
# randomly sample 3 images 
#make grid
# save in grids/unseen_coco/coco_class/web_concept 
