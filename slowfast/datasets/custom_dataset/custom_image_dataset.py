#!/usr/bin/env python3
# Created in the VideoProcessMining project

import logging
import torchvision.transforms.functional as TF
import cv2
import torch
from . import custom_helper as custom_helper


logger = logging.getLogger(__name__)


class Custom_Image(torch.utils.data.Dataset):
    """
    This dataset contains the image information
    of all images (not only key images)
    """

    def __init__(self, cfg, split):
        """
        Initialize the variables
        :param cfg: the config
        :param split: the current mode, "train" or "val"
        """
        self.cfg = cfg
        self._split = split

        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths from files and fill self._images
        as list of all images used in the dataset
        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        self._image_paths, self._video_idx_to_name = custom_helper.load_image_lists(
            cfg, is_train=(self._split == "train")
        )

        # Creating a multidimensional list of all images
        # columns: frame_path, video_name, vid_counter, image_id
        self._images = []
        vid_counter = 0
        for video in self._image_paths:
            video_name = self._video_idx_to_name[vid_counter]
            image_id = 0
            for frame_path in video:
                self._images.append([frame_path, video_name, vid_counter, image_id])
                image_id += 1
            vid_counter += 1


    def __len__(self):
        """
        Returns the total number of images in the dataset
        :return:
        """
        return len(self._images)


    def __getitem__(self, idx):
        """
        Returns a single image (C, H, W) with color pixel range [0, 1], RGB order, used for mean & std calc
        :param idx: the index for self._images
        :return:
        """

        # Get frame_path at index idx
        path = self._images[idx][0]
        # Read image of shape (H, W, C) (in BGR order) and [0,255]
        img = cv2.imread(path)

        # RGB --> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # H, W, C --> C, H, W.
        # color pixel range [0,255] -> [0,1]
        data = TF.to_tensor(img)

        return data
