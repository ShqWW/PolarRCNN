import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
        
txt_ori = "train.txt"
txt_new = "train_new.txt"
root_path = './dataset/CULane'
list_path = os.path.join(root_path, 'list', txt_ori)
out_path = os.path.join(root_path, 'list', txt_new)


with open(list_path, "r") as f:
    lines = f.readlines()
prev_lines = [lines[-1]] + lines[0:-1]

lines = [(line, prev_line) for line, prev_line in zip(lines, prev_lines)]
split_size = 800
lines_mp = [(lines[i:i+split_size], ) for i in range(0, len(lines), split_size)]

def remove(lines):
    save_lines = []
    img_path = ''
    prev_img_path = ''
    for line, prev_line in tqdm(lines):
        prev_img_path = os.path.join(root_path, prev_line.strip('\n')[1:])
        if prev_img_path == img_path:
            prev_img = img
        else:
            prev_img = cv2.imread(prev_img_path)
        img_path = os.path.join(root_path, line.strip('\n')[1:])
        img = cv2.imread(img_path)

        diff = np.abs(img.astype(np.float32) - prev_img.astype(np.float32)).sum() / (img.shape[0] * img.shape[1] * img.shape[2])
        if diff > 15:
            save_lines.append(line)
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