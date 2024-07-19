import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


img_path_list = []
data_root = './dataset/LLAMAS'
img_root_path = os.path.join(data_root, 'color_images/train/')
sub_img_path_name_list = os.listdir(img_root_path)
for sub_img_path_name in sub_img_path_name_list:
    sub_img_path = os.path.join(img_root_path, sub_img_path_name)
    img_name_list = os.listdir(sub_img_path)
    img_name_list = sorted(img_name_list)
    for img_name in img_name_list:
        img_path = os.path.join(sub_img_path, img_name)
        img_path_list.append(img_path.replace(data_root, ''))


out_path = os.path.join(data_root, 'train_new.txt')

lines = img_path_list
prev_lines = [lines[-1]] + lines[0:-1]

lines = [(line, prev_line) for line, prev_line in zip(lines, prev_lines)]
split_size = 800
lines_mp = [(lines[i:i+split_size], ) for i in range(0, len(lines), split_size)]

def remove(lines):
    save_lines = []
    img_path = ''
    prev_img_path = ''
    for line, prev_line in tqdm(lines):
        prev_img_path = os.path.join(data_root, prev_line[1:])
        if prev_img_path == img_path:
            prev_img = img
        else:
            prev_img = cv2.imread(prev_img_path)
        img_path = os.path.join(data_root, line[1:])
        img = cv2.imread(img_path)

        diff = np.abs(img.astype(np.float32) - prev_img.astype(np.float32)).sum() / (img.shape[0] * img.shape[1] * img.shape[2])
        if diff > 6:
            save_lines.append(line + '\n')
    return save_lines



if __name__ == "__main__":
    with Pool(cpu_count()) as p:
        label_list_list = p.starmap(remove, lines_mp)
    label_list_new = []
    for label_list in label_list_list:
        label_list_new += label_list

    print(len(label_list_new))
    with open(out_path, 'w') as f:
        f.writelines(label_list_new)