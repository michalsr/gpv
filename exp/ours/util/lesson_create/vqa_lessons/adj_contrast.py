import utils.io as io
import json 
import numpy as np
from tqdm import tqdm 


verb_list = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/verb_category_classification.json')
adj_list = io.load_json_object('/shared/rsaas/michal5/gpv_michal/web_training_info/adj_category_classification.json')

