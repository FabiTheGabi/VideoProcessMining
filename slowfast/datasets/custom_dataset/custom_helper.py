#!/usr/bin/env python3
# Created in the VideoProcessMining project based on the ava_helper

import logging
import os
from collections import defaultdict
from fvcore.common.file_io import PathManager

logger = logging.getLogger(__name__)



def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = [
        os.path.join(cfg.CUSTOM_DATASET.FRAME_LIST_DIR, filename)
        for filename in (
            cfg.CUSTOM_DATASET.TRAIN_LISTS if is_train else cfg.CUSTOM_DATASET.TEST_LISTS
        )
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with PathManager.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(
                    os.path.join(cfg.CUSTOM_DATASET.FRAME_DIR, row[3])
                )

    image_paths = [image_paths[i] for i in range(len(image_paths))]

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )

    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    gt_lists = cfg.CUSTOM_DATASET.TRAIN_GT_BOX_LISTS if mode == "train" else []
    pred_lists = (
        cfg.CUSTOM_DATASET.TRAIN_PREDICT_BOX_LISTS
        if mode == "train"
        else cfg.CUSTOM_DATASET.TEST_PREDICT_BOX_LISTS
    )

    if not os.path.isfile(os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, pred_lists[0])):
        pred_lists = []

    ann_filenames = [
        os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    CUSTOM_DATASET_VALID_FRAMES = range(cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND, cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND +1)

    detect_thresh = cfg.CUSTOM_DATASET.DETECTION_SCORE_THRESH
    all_boxes = {}
    count = 0
    unique_box_count = 0
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with PathManager.open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                if not is_gt_box:
                    score = float(row[7])
                    if score < detect_thresh:
                        continue

                video_name, frame_sec = row[0], int(row[1])

                # Only select frame_sec % 4 = 0 samples for validation if not
                # set FULL_TEST_ON_VAL.
                if (
                    mode == "val"
                    and not cfg.CUSTOM_DATASET.FULL_TEST_ON_VAL
                    and frame_sec % 4 != 0
                ):
                    continue

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                box_key = ",".join(row[2:6])
                box = list(map(float, row[2:6]))
                label = -1 if row[6] == "" else int(row[6])

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}
                    for sec in CUSTOM_DATASET_VALID_FRAMES:
                        all_boxes[video_name][sec] = {}

                if box_key not in all_boxes[video_name][frame_sec]:
                    all_boxes[video_name][frame_sec][box_key] = [box, []]
                    unique_box_count += 1

                all_boxes[video_name][frame_sec][box_key][1].append(label)
                if label != -1:
                    count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    logger.info(
        "Finished loading annotations from: %s" % ", ".join(ann_filenames)
    )
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels, cfg):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec, cfg):
        """
        Convert time index (in second) to frame index.
        [Exemplary values with start second 900, format
        frame_index: time_in_seconds from video start]
        0: 900
        30: 901
        """
        start_second = cfg.PREPROCESS.START_SECOND if cfg.PREPROCESS.START_SECOND > -1 else 0
        return (sec - start_second) * cfg.CUSTOM_DATASET.FRAME_RATE

    CUSTOM_DATASET_VALID_FRAMES = range(cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND,
                                        cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND +1)

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            if sec not in CUSTOM_DATASET_VALID_FRAMES:
                continue

            if len(boxes_and_labels[video_idx][sec]) > 0:
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec_to_frame(sec, cfg))
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count
