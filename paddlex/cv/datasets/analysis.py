# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import numpy as np
import os.path as osp
import cv2
from PIL import Image
import pickle
import threading
import multiprocessing as mp

import paddlex.utils.logging as logging
from paddlex.utils import path_normalization
from paddlex.cv.transforms.seg_transforms import Compose
from .dataset import get_encoding


class Seg:
    def __init__(self, data_dir, file_list, label_list):
        self.data_dir = data_dir
        self.file_list_path = file_list
        self.file_list = []
        self.labels = []
        with open(label_list, encoding=get_encoding(label_list)) as f:
            for line in f:
                item = line.strip()
                self.labels.append(item)

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                if len(items) > 2:
                    raise Exception(
                        f"A space is defined as the separator, but it exists in image or label name {line}."
                    )
                items[0] = path_normalization(items[0])
                items[1] = path_normalization(items[1])
                full_path_im = osp.join(data_dir, items[0])
                full_path_label = osp.join(data_dir, items[1])
                if not osp.exists(full_path_im):
                    raise IOError(f'The image file {full_path_im} is not exist!')
                if not osp.exists(full_path_label):
                    raise IOError(f'The image file {full_path_label} is not exist!')
                self.file_list.append([full_path_im, full_path_label])
        self.num_samples = len(self.file_list)

    def _get_shape(self):
        max_height = max(self.im_height_list)
        max_width = max(self.im_width_list)
        min_height = min(self.im_height_list)
        min_width = min(self.im_width_list)
        return {
            'max_height': max_height,
            'max_width': max_width,
            'min_height': min_height,
            'min_width': min_width,
        }

    def _get_label_pixel_info(self):
        pixel_num = np.dot(self.im_height_list, self.im_width_list)
        label_pixel_info = {}
        for label_value, label_value_num in zip(self.label_value_list,
                                                self.label_value_num_list):
            for v, n in zip(label_value, label_value_num):
                if v in label_pixel_info:
                    label_pixel_info[v][0] += n
                    label_pixel_info[v][1] += float(n) / float(pixel_num)

                else:
                    label_pixel_info[v] = [n, float(n) / float(pixel_num)]
        return label_pixel_info

    def _get_image_pixel_info(self):
        channel = max(len(im_value) for im_value in self.im_value_list)
        im_pixel_info = [{} for _ in range(channel)]
        for im_value, im_value_num in zip(self.im_value_list,
                                          self.im_value_num_list):
            for c in range(channel):
                for v, n in zip(im_value[c], im_value_num[c]):
                    if v not in im_pixel_info[c].keys():
                        im_pixel_info[c][v] = n
                    else:
                        im_pixel_info[c][v] += n
        return im_pixel_info

    def _get_mean_std(self):
        im_mean = np.asarray(self.im_mean_list)
        im_mean = im_mean.sum(axis=0)
        im_mean = im_mean / len(self.file_list)
        im_mean /= self.max_im_value - self.min_im_value

        im_std = np.asarray(self.im_std_list)
        im_std = im_std.sum(axis=0)
        im_std = im_std / len(self.file_list)
        im_std /= self.max_im_value - self.min_im_value

        return (im_mean, im_std)

    def _get_image_info(self, start, end):
        for id in range(start, end):
            full_path_im, full_path_label = self.file_list[id]
            image, label = Compose.decode_image(full_path_im, full_path_label)

            height, width, channel = image.shape
            self.im_height_list[id] = height
            self.im_width_list[id] = width
            self.im_channel_list[id] = channel

            self.im_mean_list[
                id] = [image[:, :, c].mean() for c in range(channel)]
            self.im_std_list[
                id] = [image[:, :, c].std() for c in range(channel)]
            for c in range(channel):
                unique, counts = np.unique(image[:, :, c], return_counts=True)
                self.im_value_list[id].extend([unique])
                self.im_value_num_list[id].extend([counts])

            unique, counts = np.unique(label, return_counts=True)
            self.label_value_list[id] = unique
            self.label_value_num_list[id] = counts

    def _get_clipped_mean_std(self, start, end, clip_min_value, clip_max_value):
        for id in range(start, end):
            full_path_im, full_path_label = self.file_list[id]
            image, label = Compose.decode_image(full_path_im, full_path_label)
            for c in range(self.channel_num):
                np.clip(
                    image[:, :, c],
                    clip_min_value[c],
                    clip_max_value[c],
                    out=image[:, :, c])
                image[:, :, c] -= clip_min_value[c]
                image[:, :, c] /= clip_max_value[c] - clip_min_value[c]
            self.clipped_im_mean_list[id] = [
                image[:, :, c].mean() for c in range(self.channel_num)
            ]
            self.clipped_im_std_list[
                id] = [image[:, :, c].std() for c in range(self.channel_num)]

    def analysis(self):
        self.im_mean_list = [[] for _ in range(len(self.file_list))]
        self.im_std_list = [[] for _ in range(len(self.file_list))]
        self.im_value_list = [[] for _ in range(len(self.file_list))]
        self.im_value_num_list = [[] for _ in range(len(self.file_list))]
        self.im_height_list = np.zeros(len(self.file_list), dtype='int64')
        self.im_width_list = np.zeros(len(self.file_list), dtype='int64')
        self.im_channel_list = np.zeros(len(self.file_list), dtype='int64')
        self.label_value_list = [[] for _ in range(len(self.file_list))]
        self.label_value_num_list = [[] for _ in range(len(self.file_list))]

        num_workers = min(mp.cpu_count() // 2, 8)
        threads = []
        one_worker_file = len(self.file_list) // num_workers
        for i in range(num_workers):
            start = one_worker_file * i
            end = one_worker_file * (
                i + 1) if i < num_workers - 1 else len(self.file_list)
            t = threading.Thread(target=self._get_image_info, args=(start, end))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        unique, counts = np.unique(self.im_channel_list, return_counts=True)
        if len(unique) > 1:
            raise Exception(
                f"There are {len(unique)} kinds of image channels: {unique[:]}."
            )
        self.channel_num = unique[0]
        shape_info = self._get_shape()
        self.max_height = shape_info['max_height']
        self.max_width = shape_info['max_width']
        self.min_height = shape_info['min_height']
        self.min_width = shape_info['min_width']
        self.label_pixel_info = self._get_label_pixel_info()
        self.im_pixel_info = self._get_image_pixel_info()
        mode = osp.split(self.file_list_path)[-1].split('.')[0]
        import matplotlib.pyplot as plt
        for c in range(self.channel_num):
            plt.figure()
            plt.bar(self.im_pixel_info[c].keys(),
                    self.im_pixel_info[c].values(),
                    width=1,
                    log=True)
            plt.xlabel('image pixel value')
            plt.ylabel('number')
            plt.title(f'channel={c}')
            plt.savefig(
                osp.join(self.data_dir, f'{mode}_channel{c}_distribute.png'),
                dpi=100,
            )
            plt.close()

        max_im_value = []
        min_im_value = []
        for c in range(self.channel_num):
            max_im_value.append(max(self.im_pixel_info[c].keys()))
            min_im_value.append(min(self.im_pixel_info[c].keys()))
        self.max_im_value = np.asarray(max_im_value)
        self.min_im_value = np.asarray(min_im_value)

        im_mean, im_std = self._get_mean_std()

        info = {
            'channel_num': self.channel_num,
            'image_pixel': self.im_pixel_info,
            'label_pixel': self.label_pixel_info,
            'file_num': len(self.file_list),
            'max_height': self.max_height,
            'max_width': self.max_width,
            'min_height': self.min_height,
            'min_width': self.min_width,
            'max_image_value': self.max_im_value,
            'min_image_value': self.min_im_value
        }
        saved_pkl_file = osp.join(self.data_dir, f'{mode}_infomation.pkl')
        with open(osp.join(saved_pkl_file), 'wb') as f:
            pickle.dump(info, f)

        logging.info(
            "############## The analysis results are as follows ##############\n"
        )
        logging.info(f"{len(self.file_list)} samples in file {self.file_list_path}\n")
        logging.info(
            f"Minimal image height: {self.min_height} Minimal image width: {self.min_width}.\n"
        )
        logging.info(
            f"Maximal image height: {self.max_height} Maximal image width: {self.max_width}.\n"
        )
        logging.info(f"Image channel is {self.channel_num}.\n")
        logging.info(
            f"Minimal image value: {self.min_im_value} Maximal image value: {self.max_im_value} (arranged in 0-{self.channel_num} channel order) \n"
        )
        logging.info(
            f"Image pixel distribution of each channel is saved with 'distribute.png' in the {self.data_dir}"
        )
        logging.info(
            f"Image mean value: {im_mean} Image standard deviation: {im_std} (normalized by the (max_im_value - min_im_value), arranged in 0-{self.channel_num} channel order).\n"
        )
        logging.info(
            "Label pixel information is shown in a format of (label_id, the number of label_id, the ratio of label_id):"
        )
        for v, (n, r) in self.label_pixel_info.items():
            logging.info(f"({v}, {n}, {r})")

        logging.info(f"Dataset information is saved in {saved_pkl_file}")

    def cal_clipped_mean_std(self, clip_min_value, clip_max_value,
                             data_info_file):
        if not osp.exists(data_info_file):
            raise Exception(f"Dataset information file {data_info_file} does not exist.")
        with open(data_info_file, 'rb') as f:
            im_info = pickle.load(f)
        channel_num = im_info['channel_num']
        min_im_value = im_info['min_image_value']
        max_im_value = im_info['max_image_value']
        im_pixel_info = im_info['image_pixel']

        if len(clip_min_value) != channel_num or len(
                clip_max_value) != channel_num:
            raise Exception(
                f"The length of clip_min_value or clip_max_value should be equal to the number of image channel {channle_num}."
            )
        for c in range(channel_num):
            if clip_min_value[c] < min_im_value[c] or clip_min_value[
                    c] > max_im_value[c]:
                raise Exception(
                    f"Clip_min_value of the channel {c} is not in [{min_im_value[c]}, {max_im_value[c]}]"
                )
            if clip_max_value[c] < min_im_value[c] or clip_max_value[
                    c] > max_im_value[c]:
                raise Exception(
                    f"Clip_max_value of the channel {c} is not in [{min_im_value[c]}, {self.max_im_value[c]}]"
                )

        self.clipped_im_mean_list = [[] for _ in range(len(self.file_list))]
        self.clipped_im_std_list = [[] for _ in range(len(self.file_list))]

        num_workers = min(mp.cpu_count() // 2, 8)
        threads = []
        one_worker_file = len(self.file_list) // num_workers
        self.channel_num = channel_num
        for i in range(num_workers):
            start = one_worker_file * i
            end = one_worker_file * (
                i + 1) if i < num_workers - 1 else len(self.file_list)
            t = threading.Thread(
                target=self._get_clipped_mean_std,
                args=(start, end, clip_min_value, clip_max_value))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        im_mean = np.asarray(self.clipped_im_mean_list)
        im_mean = im_mean.sum(axis=0)
        im_mean = im_mean / len(self.file_list)

        im_std = np.asarray(self.clipped_im_std_list)
        im_std = im_std.sum(axis=0)
        im_std = im_std / len(self.file_list)

        for c in range(channel_num):
            pixel_num = sum(im_pixel_info[c].values())
            clip_pixel_num = sum(
                n
                for v, n in im_pixel_info[c].items()
                if v < clip_min_value[c] or v > clip_max_value[c]
            )
            logging.info(
                f"Channel {c}, the ratio of pixels to be clipped = {clip_pixel_num / pixel_num}"
            )

        logging.info(
            f"Image mean value: {im_mean} Image standard deviation: {im_std} (normalized by (clip_max_value - clip_min_value), arranged in 0-{self.channel_num} channel order).\n"
        )
