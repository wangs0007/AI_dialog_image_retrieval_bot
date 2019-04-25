import torch
import os
import json
import cv2

with open('name_path.json', 'r') as load_f:
    load_dict = json.load(load_f)

test_path = '/home/wangshuo/work/project/fashion-retrieval/eval_imgs'

with open('dataset/eval_im_names.txt') as f:
    idx = 0
    while (True):
        im = f.readline().strip()
        if im == '':
            break
        src = load_dict[im]
        dst = os.path.join(test_path, str(idx) + '.jpg')
        idx += 1
        img = cv2.imread(src)
        cv2.imwrite(dst, img)

