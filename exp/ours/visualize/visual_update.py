import os 
import PIL.Image
import numpy as np
import imagesize
from PIL import Image
from exp.ours.util import py_utils, image_utils
from utils import box_ops
import skimage.draw as skdraw
from PIL import Image
import torch
import torchvision.ops
import utils.io as io 
def cxcywh_to_xyxy(bbox,im_h=1,im_w=1):
    cx,cy,w,h = bbox
    cx,cy,w,h = cx*im_w,cy*im_h,w*im_w,h*im_h
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    return (x1,y1,x2,y2)
def vis_bbox(bbox,img,color=(255,0,0),modify=False,alpha=0.2,fmt='ncxcywh'):
    im_h,im_w = img.shape[0:2]
    if fmt=='cxcywh':
        bbox = cxcywh_to_xyxy(bbox)
    elif fmt=='ncxcywh':
        bbox = cxcywh_to_xyxy(bbox,im_h,im_w)
    elif fmt=='xyxy':
        pass
    else:
        raise NotImplementedError(f'fmt={fmt} not implemented')

    x1,y1,x2,y2 = bbox
    x1 = max(0,min(x1,im_w-1))
    x2 = max(x1,min(x2,im_w-1))
    y1 = max(0,min(y1,im_h-1))
    y2 = max(y1,min(y2,im_h-1))
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]

    if modify==True:
        img_ = img
    else:
        img_ = np.copy(img)

    if len(img.shape)==2:
        color = (color[0],)

    rr,cc = skdraw.polygon(r,c,img.shape[:2])
    skdraw.set_color(img_,(rr,cc),color,alpha=alpha)

    rr,cc = skdraw.polygon_perimeter(r,c,img.shape[:2])
    
    if len(img.shape)==3:
        for k in range(3):
            img_[rr,cc,k] = color[k]
    elif len(img.shape)==2:
        img_[rr,cc]=color[0]

    return img_

def make_image_html(x1, y1, x2, y2, rel=None, rank=None, color="black", border_width="medium"):
  rect_style = {
    "position": "absolute",
    "top": y1,
    "left": x1,
    "height": y2-y1,
    "width": x2-x1,
    "border-style": "solid",
    "border-color": color,
    "border-width": border_width,
    "box-sizing": "border-box"
  }
  rect_style_str = "; ".join(f"{k}: {v}" for k, v in rect_style.items())

  text_style = {
    "position": "absolute",
    "top": y1-5,
    "left": x1+3,
    "color": color,
    "background-color": "black",
    "z-index": 9999,
    "padding-right": "5px",
    "padding-left": "5px",
  }
  text_style_str = "; ".join(f"{k}: {v}" for k, v in text_style.items())

  if rel is None and rank is None:
    container = ""
  else:
    container = f'class=box'
    if rel:
      container += f' data-rel="{rel}"'
    if rank:
      container += f' data-rank="{rank}"'

  html = [
    f'<div {container}>',
    f'  <div style="{rect_style_str}"></div>',
    ('' if rel is None else f'  <div style="{text_style_str}">{rel:0.2f}</div>') +
    "</div>"
  ]
  return html
def _html(tag, prod, style=""):
  return f'<{tag} style="{style}">{prod}</{tag}>'

def create_img_entry(image_src,actual_img_source):
    html = []
    html += [f'<div style="display: inline-block; position: relative;">']
    image_w, image_h = image_utils.get_image_size(image_src)
    image_attr = dict(src=actual_img_source)
    print(actual_img_source)
    attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
    html += [f'<img src={actual_img_source} height={image_h} width={image_w}>']
   
    # html += make_image_html(
    #         x1*w_factor, y1*h_factor, x2*w_factor, y2*h_factor)
    return html 

def get_table_html(rows):
  html = []
  style = """
table td {
border: thin solid; 
}
table th {
border: thin solid;
}
  """
  print(rows[0],'first row')
  html.append("<style>")
  html.append(style)
  html.append("</style>")

  html += ["<div>"]
  html += ['<table style="font-size:20px; margin-left: auto; margin-right: auto; border-collapse: collapse;">']
 

  
  #cols = ['Initial','Hierarchical', 'Contrast','MIL','Query','Score']

  #cols = ['MIL','Query','Score']
  cols = ['Hierarchical','Query','Score']
  html += ['\t<tr>']
  for col in cols:
    html += [_html("th", col, "text-align:center")]
  html += ["\t</tr>"]

  for row in rows:
    html += [f'\t<tr>']
    for k in cols:
      html += [f'<td style="text-align:center">']
      val = [""] if k not in row else row[k]

    
      if isinstance(val, list):
        html += val
      else:
        html.append(str(val))
      html += ["</td>"]
    html += ["\t</tr>"]
  html += ["</table>"]
  html += ["</div>"]

  return html
def create_html_file(entries,file_name):
    table = []
    for entry in entries:
        row = dict()
        #row['Initial'] = create_img_entry(entry['initial'],entry['initial_actual_img_src'])
        #row['MIL'] = create_img_entry(entry['MIL'],entry['MIL_actual_img_src'])
        #row['Contrast'] = create_img_entry(entry['contrast'],entry['contrast_actual_img_src'])
        row['Hierarchical'] = create_img_entry(entry['hierarchical'],entry['hierarchical_actual_img_src'])
        #row['MIL'] = create_img_entry(entry['mil'],entry['mil_actual_img_src'])
        
        row['Query'] = entry['rel_query']
        row['Score'] = entry['rel']

        table.append(row)

    final_table = get_table_html(table)
    html = "\n".join(final_table)
    with open(file_name, "w") as f:
        f.write(html)
def create_visual_outputs(model_name,number,model_output):
  image_file = image_utils.get_image_file(model_output['image_ids'][number])
  img = Image.open(image_file)
  print(model_output['pred_boxes'][number])
  #box = torchvision.ops.box_convert(torch.tensor(model_output['pred_boxes'][number]),"cxcywh", "cxcywh")
  updated_img_np = vis_bbox(model_output['pred_boxes'][number],np.asarray(img),modify=True)
  updated_img = Image.fromarray(updated_img_np)
  updated_img = updated_img.resize((224, 224), Image.ANTIALIAS)
  if not os.path.exists(f'/home/michal/gpv_michal/outputs/hierarchical_vis_html/html_file/{model_name}'):
        os.mkdir(f'/home/michal/gpv_michal/outputs/hierarchical_vis_html/html_file/{model_name}')
  updated_img.save(f'/home/michal/gpv_michal/outputs/hierarchical_vis_html/html_file/{model_name}/img_{number}.jpg')
  return f'/home/michal/gpv_michal/outputs/hierarchical_vis_html/html_file/{model_name}/img_{number}.jpg',f'{model_name}/img_{number}.jpg'
def make_html_entries():
  final_entries = []
  #make dict mapping from col name to output 
  #add 3 entries entry['hiearchical'] = image, entry['contrast'] = image, etc
  #load model output from each model 
  #choose 5 coco entries to do 
  #after map top 5 bounding boxes for each model  
  initial_output = io.load_json_object('/home/michal/gpv_michal/outputs/initial_unseen_group_1_final/vis_pred/vis_data.json')
  mil_output = io.load_json_object('/home/michal/gpv_michal/outputs/mil_unseen_group_1_final/vis_pred/vis_data.json')
  contrast_output = io.load_json_object('/home/michal/gpv_michal/outputs/contrast_unseen_group_1_final/vis_pred/vis_data.json')
  syn_output = io.load_json_object('/home/michal/gpv_michal/outputs/syn_unseen_group_1_final/vis_pred/vis_data.json')
  for i in range(len(initial_output['images'])):
    entry = {}
    entry['initial'],entry['initial_actual_img_src'] = create_visual_outputs('initial',i,initial_output)
    entry['MIL'],entry['MIL_actual_img_src'] = create_visual_outputs('mil',i,mil_output)
    entry['contrast'],entry['contrast_actual_img_src'] = create_visual_outputs('contrast',i,contrast_output)
    entry['hierarchical'],entry['hierarchical_actual_img_src'] = create_visual_outputs('hierarchical',i,syn_output)
    entry['rel_query'] = initial_output['queries'][i]

    final_entries.append(entry)
    print(entry,'entry')
  io.dump_json_object(final_entries,f'/home/michal/gpv_michal/outputs/unseen_group_1_html/html_file/html_entries.json')
  return final_entries

def second_html_entries():
  final_entries = []
  new_syn_vis_data = io.load_json_object('/home/michal/gpv_michal/outputs/syn_vis_data/html_files.json')

  #for i in range(len(new_syn_vis_data['queries'])):
  print(len(new_syn_vis_data['image_ids']),'num')
  for i in range(1000):
    entry = {}
    #entry['mil'],entry['mil_actual_img_src'] = create_visual_outputs('mil',i,new_syn_vis_data)
    entry['hierarchical'],entry['hierarchical_actual_img_src'] = create_visual_outputs('hierarchical',i,new_syn_vis_data)
    entry['rel_query'] = new_syn_vis_data['queries'][i]
    entry['rel'] = new_syn_vis_data['rel'][i]
    final_entries.append(entry)
  io.dump_json_object(final_entries,'/home/michal/gpv_michal/outputs/hierarchical_vis_html/html_entries.json')
  return final_entries

entries = second_html_entries()

create_html_file(entries,'/home/michal/gpv_michal/outputs/hierarchical_vis_html/html_file/vis.html')



