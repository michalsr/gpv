import os 
import PIL.Image

import imagesize
from PIL import Image
from exp.ours.util import py_utils, image_utils
from utils import box_ops
import utils.io as io 



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

def create_img_entry(image_src,box,actual_img_source):
    html = []
    html += [f'<div style="display: inline-block; position: relative;">']
    image_w, image_h = image_utils.get_image_size(image_src)
    image_attr = dict(src=actual_img_source)
    print(actual_img_source)
    attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
    html += [f'<img src={actual_img_source} height={image_h} width={image_w}>']
    w_factor = image_w
    h_factor = image_h
    x1, y1, x2, y2 = box
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
  html.append("<style>")
  html.append(style)
  html.append("</style>")

  html += ["<div>"]
  html += ['<table style="font-size:20px; margin-left: auto; margin-right: auto; border-collapse: collapse;">']
  print(rows)
  all_keys = rows[0]
  for row in rows[1:]:
    all_keys.update(row)
  cols = list(all_keys)

  html += ['\t<tr>']
  for col in cols:
    html += [_html("th", col, "text-align:center")]
  html += ["\t</tr>"]

  for row in rows:
    html += [f'\t<tr>']
    for k in all_keys:
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
        row["image"] = create_img_entry(entry['img_src'], entry['box'],entry['actual_img_src'])
        row['rel_query'] = entry['rel_query']
        row['model'] = entry['model']
        table.append(row)
    print(table)
    final_table = get_table_html(table)
    html = "\n".join(final_table)
    with open(file_name, "w") as f:
        f.write(html)


entries = io.load_json_object('/home/michal/gpv_michal/html_file/html_entries.json')
print(entries)
create_html_file(entries,'/home/michal/gpv_michal/html_file/vis.html')



