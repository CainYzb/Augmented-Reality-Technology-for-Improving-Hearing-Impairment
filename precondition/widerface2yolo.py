# author: wujiahao
# Created: 2022/3/14
#       convert widerface to yolo format

import os
import cv2
import shutil

data_root = './WIDER_train/images' 
ann_txt_path = r'C:\Users\17845\Desktop\Code\SoftwareEngineer\yolov7-face-master\wider_face_split\wider_face_train_bbx_gt.txt'
dst_path = './wider2yolo'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
    os.mkdir(os.path.join(dst_path, 'images'))
    os.mkdir(os.path.join(dst_path, 'labels'))


with open(ann_txt_path, 'r') as f:
    ann_lines = f.readlines()

line_i = 0
img_i = 0
while line_i < len(ann_lines):
    # every img for every iter
    img_path = ann_lines[line_i].strip()
    print(f'{img_i}  {line_i} : {img_path}')
    img = cv2.imread(os.path.join(data_root, img_path))
    h_img, w_img, c_img = img.shape

    # convert ann
    line_i += 1
    num_bbox = int(ann_lines[line_i].strip())

    # empty bbox
    if num_bbox ==0 : 
        line_i += 2
        img_i += 1
        continue
    
    out_lines = [] # lines write to yolo txt
    for bbox_i in range(num_bbox):
        # every bbox for every iter
        line_i += 1
        bbox_line = ann_lines[line_i].strip() 
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = [int(i) for i in bbox_line.split()]
        out_lines.append(f'{0} {(x1+w/2)/w_img} {(y1+h/2)/h_img} {w/w_img} {h/h_img}\n')
    # write to yolo txt    
    with open(os.path.join(dst_path, 'labels', f"{img_i}.txt"), 'w') as f:
        f.writelines(out_lines)
    # copy img
    shutil.copy(os.path.join(data_root, img_path), os.path.join(dst_path, 'images', f"{img_i}.{img_path.split('.')[-1]}"))
    # next img
    line_i +=1
    img_i += 1

