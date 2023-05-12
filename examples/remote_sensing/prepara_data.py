import os
import os.path as osp
import numpy as np
import cv2
import shutil
from PIL import Image
import paddlex as pdx

# 定义训练集切分时的滑动窗口大小和步长，格式为(W, H)
train_tile_size = (1024, 1024)
train_stride = (512, 512)
# 定义验证集切分时的滑动窗口大小和步长，格式(W, H)
val_tile_size = (769, 769)
val_stride = (769, 769)

# 下载并解压2015 CCF大数据比赛提供的高清遥感影像
ccf_remote_dataset = 'https://bj.bcebos.com/paddlex/examples/remote_sensing/datasets/ccf_remote_dataset.tar.gz'
pdx.utils.download_and_decompress(ccf_remote_dataset, path='./')

if not osp.exists('./dataset/JPEGImages'):
    os.makedirs('./dataset/JPEGImages')
if not osp.exists('./dataset/Annotations'):
    os.makedirs('./dataset/Annotations')

# 将前4张图片划分入训练集，并切分成小块之后加入到训练集中
# 并生成train_list.txt
for train_id in range(1, 5):
    shutil.copyfile(
        f"ccf_remote_dataset/{train_id}.png",
        f"./dataset/JPEGImages/{train_id}.png",
    )
    shutil.copyfile(
        f"ccf_remote_dataset/{train_id}_class.png",
        f"./dataset/Annotations/{train_id}_class.png",
    )
    mode = 'w' if train_id == 1 else 'a'
    with open('./dataset/train_list.txt', mode) as f:
        f.write(f"JPEGImages/{train_id}.png Annotations/{train_id}_class.png\n")

for train_id in range(1, 5):
    image = cv2.imread(f'ccf_remote_dataset/{train_id}.png')
    label = Image.open(f'ccf_remote_dataset/{train_id}_class.png')
    H, W, C = image.shape
    train_tile_id = 1
    for h in range(0, H, train_stride[1]):
        for w in range(0, W, train_stride[0]):
            left = w
            upper = h
            right = min(w + train_tile_size[0] * 2, W)
            lower = min(h + train_tile_size[1] * 2, H)
            tile_image = image[upper:lower, left:right, :]
            cv2.imwrite(f"./dataset/JPEGImages/{train_id}_{train_tile_id}.png", tile_image)
            cut_label = label.crop((left, upper, right, lower))
            cut_label.save(f"./dataset/Annotations/{train_id}_class_{train_tile_id}.png")
            with open('./dataset/train_list.txt', 'a') as f:
                f.write(
                    f"JPEGImages/{train_id}_{train_tile_id}.png Annotations/{train_id}_class_{train_tile_id}.png\n"
                )
            train_tile_id += 1

# 将第5张图片切分成小块之后加入到验证集中
val_id = 5
val_tile_id = 1
shutil.copyfile(
    f"ccf_remote_dataset/{val_id}.png", f"./dataset/JPEGImages/{val_id}.png"
)
shutil.copyfile(
    f"ccf_remote_dataset/{val_id}_class.png",
    f"./dataset/Annotations/{val_id}_class.png",
)
image = cv2.imread(f'ccf_remote_dataset/{val_id}.png')
label = Image.open(f'ccf_remote_dataset/{val_id}_class.png')
H, W, C = image.shape
for h in range(0, H, val_stride[1]):
    for w in range(0, W, val_stride[0]):
        left = w
        upper = h
        right = min(w + val_tile_size[0], W)
        lower = min(h + val_tile_size[1], H)
        cut_image = image[upper:lower, left:right, :]
        cv2.imwrite(f"./dataset/JPEGImages/{val_id}_{val_tile_id}.png", cut_image)
        cut_label = label.crop((left, upper, right, lower))
        cut_label.save(f"./dataset/Annotations/{val_id}_class_{val_tile_id}.png")
        mode = 'w' if val_tile_id == 1 else 'a'
        with open('./dataset/val_list.txt', mode) as f:
            f.write(
                f"JPEGImages/{val_id}_{val_tile_id}.png Annotations/{val_id}_class_{val_tile_id}.png\n"
            )
        val_tile_id += 1

# 生成labels.txt
label_list = ['background', 'vegetation', 'road', 'building', 'water']
for i, label in enumerate(label_list):
    mode = 'w' if i == 0 else 'a'
    with open('./dataset/labels.txt', 'a') as f:
        name = f"{label}\n" if i < len(label_list) - 1 else f"{label}"
        f.write(name)
