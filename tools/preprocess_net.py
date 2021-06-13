# Created in the VideoProcessMining project

import csv
import logging
import sys
import numpy as np
from shutil import copyfile

import cv2
import torch
from torch.utils.data import DataLoader

from slowfast.datasets import build_dataset
from slowfast.datasets.custom_dataset.preprocessing.detectron2_object_predictor_preprocess import \
    PreprocessDetectron2ObjectPredictor
from slowfast.datasets.custom_dataset.custom_image_dataset import Custom_Image
from slowfast.utils.env import setup_environment
import os
from pathlib import Path
import mimetypes

import pandas as pd

logger = logging.getLogger(__name__)



def create_folder_structure(cfg, progress_callback=None):
    """
    Creates an AVA like folder structure for a custom dataset at
    cfg.OUTPUT_DIR
    :param cfg: config
    """

    # Create root folder
    create_folder(cfg.OUTPUT_DIR)

    # Create sub folders
    sub_folder_names = ["annotations", "checkpoints", "demo", "frame_lists", "frames", "runs-custom", "videos"]
    for sub_folder in sub_folder_names:
        sub_dir = os.path.join(cfg.OUTPUT_DIR, sub_folder)
        create_folder(sub_dir)

    # Copy files into annotation folder
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    copyfile(os.path.join(ROOT_DIR, "slowfast/datasets/custom_dataset/exemplary_annotation_files/train_groundtruth.csv"), os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.TRAIN_GT_BOX_LISTS[0]))
    copyfile(os.path.join(ROOT_DIR, "slowfast/datasets/custom_dataset/exemplary_annotation_files/val_groundtruth.csv"), os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.GROUNDTRUTH_FILE))
    copyfile(os.path.join(ROOT_DIR, "slowfast/datasets/custom_dataset/exemplary_annotation_files/label_map_file.pbtxt"), os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.LABEL_MAP_FILE))

    # Insert empty files for the rest
    empty_file_list_names = [cfg.CUSTOM_DATASET.TEST_PREDICT_BOX_LISTS[0], cfg.CUSTOM_DATASET.TRAIN_PREDICT_BOX_LISTS[0],
                             cfg.CUSTOM_DATASET.EXCLUSION_FILE]
    for file in empty_file_list_names:
        file_paths = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, file)
        with open(file_paths, "w") as my_empty_csv:
            # now you have an empty file already
            pass  # or write something to it already

    logger.info("Created root folder at: %s" % cfg.OUTPUT_DIR)


def write_valid_frames_conform_csv(cfg, path_to_csvs_list, csv_dst_dir_path, middle_frame_timestamp_column,
                                   header_type=None):
    """
    Concats multiple csvs, filters out only valid frames, and
        exports single combined and filtered file to dst_dir
    :param cfg: the config
    :param path_to_csvs_list: a list of paths to csv files that are combined
    :param csv_dst_dir_path: the path to the newly created csv file
    :param middle_frame_timestamp_column: the column that contains the middle_frame_timestamp
    :param: header_type of the csv files
    :return:
    """

    # Concats the csv files
    new_csv = pd.concat(
        [pd.read_csv(f, header=header_type, float_precision='high') for f in path_to_csvs_list if
         os.path.isfile(f)])

    # Filter out rows with non relevant middle_frame_timestamps
    new_csv = new_csv[
        (new_csv[new_csv.columns[middle_frame_timestamp_column]] >= cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND)
        & (new_csv[new_csv.columns[middle_frame_timestamp_column]] <= cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND)]

    # Export to csv
    new_csv.to_csv(csv_dst_dir_path, index=False, header=header_type)


def extract_frames_from_gt_videos(cfg, gt_video_names_unique):
    """
    For every input video a sub directory in the frames folder is created
    and the images are extracted at framerate fps.
    It is possible to cut the videos from start_second to end_second
    |_ frames
    |  |_ [video name 0]
    |  |  |_ [video name 0]_0000001.jpg
    |  |  |_ [video name 0]_0000002.jpg
    :param cfg: the config file
    :param gt_video_names_unique: the unique names of videos in all GT files
    :return:
    """

    video_files = os.listdir(cfg.PREPROCESS.ORIGINAL_VIDEO_DIR) if os.path.exists(
        cfg.PREPROCESS.ORIGINAL_VIDEO_DIR) else []

    video_file_names = [Path(video).stem for video in video_files]

    if len(video_files) > 0:
        for video, video_name in zip(video_files, video_file_names):
            if is_gt_video_file(video, gt_video_names_unique):
                cut_command = ""
                # Create new subfolder for each video
                sub_dir = os.path.join(cfg.CUSTOM_DATASET.FRAME_DIR, video_name)
                create_folder(sub_dir)

                # Cut if required
                if cfg.PREPROCESS.START_SECOND >= 0 and cfg.PREPROCESS.END_SECOND > 0:
                    cut_command = " -ss " + str(cfg.PREPROCESS.START_SECOND) + " -t " + str(
                        cfg.PREPROCESS.END_SECOND - cfg.PREPROCESS.START_SECOND)

                # Extract frames into new folder
                input_file = os.path.join(cfg.PREPROCESS.ORIGINAL_VIDEO_DIR, video)
                ffmpeg_command = "ffmpeg -loglevel panic" + cut_command + " -i " + input_file + " -r " + str(
                    cfg.CUSTOM_DATASET.FRAME_RATE) + \
                                 " -q:v 1 " + sub_dir + "/" + video_name + "_%07d.jpg"
                os.system(ffmpeg_command)
                """
                Explanation for the ffmpeg command:
                -loglevel panic: show only serious errors
                -ss: seek in the input file to position. Since this is not
                    always possible, ffmpeg seeks to closes point before position
                -t: limits the time duration of data read from input file
                -i: input url
                -r: sets the frame rate
                -q:v n (same as -qscale:v n): where n is anumber from 1-31 
                    and 1 being the highest quality/largest filesize
                """

    no_video_file_for_gt_entry = list(set(gt_video_names_unique) - set(video_file_names))

    if no_video_file_for_gt_entry:
        logger.warning(
            "Attention: no video-files for gt entries: %s" % ", ".join(no_video_file_for_gt_entry)
        )

    logger.info("Extracted frames from groundtruth videos at " + str(
        cfg.CUSTOM_DATASET.FRAME_RATE) + " FPS at folder" + cfg.CUSTOM_DATASET.FRAME_DIR)


def get_unique_gt_video_information(cfg):
    """
    Extracts the unique video_ID (=video_name) for all GT videos that are
        preprocessed. The extraction is done based on the preprocessed groundtruth_files
    :param cfg:
    :param is_train: process training data
    :param is_val: process test data
    :return:
    train_gt_video_names_unique: the unique & sorted names of all videos used in train gt
    valtest_gt_video_names_unique: the unique & sorted names of all videos used in valtest gt
    """

    # The lists for the unique video_names
    train_gt_video_names_unique = []
    valtest_gt_video_names_unique = []

    # The paths to the preprocessed gt files
    train_gt_src_dir = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.TRAIN_GT_BOX_LISTS[0])
    valtest_gt_src_dir = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.GROUNDTRUTH_FILE)

    # We only want to select the video_name = first column
    columns = [0]

    # Get all video_names for all GT-files
    if os.path.isfile(train_gt_src_dir):
        train_video_names = pd.read_csv(train_gt_src_dir, header=None, usecols=columns)  # Read first column of csv file
        train_gt_video_names_unique = sorted(train_video_names[train_video_names.columns[0]].unique().tolist())

    if os.path.isfile(valtest_gt_src_dir):
        valtest_video_names = pd.read_csv(valtest_gt_src_dir, header=None, usecols=columns)
        valtest_gt_video_names_unique = sorted(valtest_video_names[valtest_video_names.columns[0]].unique().tolist())

    return train_gt_video_names_unique, valtest_gt_video_names_unique


def create_framelist_files(cfg, train_gt_video_names_unique, valtest_gt_video_names_unique):
    """
    Creates framelist files vor training and validation based on the extracted frames
    ToDo: potential improvement could be to check whether a frame is really an image
    :param cfg: the config file
    :param gt_video_names_unique: list of unique video_ID (=video_name)
    :param is_train_video:
    :return:
    """

    # The artificially created video_idx and frame_idx
    video_idx = 0

    # The output files
    framelist_train_csv_path = os.path.join(cfg.CUSTOM_DATASET.FRAME_LIST_DIR, cfg.CUSTOM_DATASET.TRAIN_LISTS[0])
    framelist_val_csv_path = os.path.join(cfg.CUSTOM_DATASET.FRAME_LIST_DIR, cfg.CUSTOM_DATASET.TEST_LISTS[0])

    with open(framelist_train_csv_path, "w") as framelist_train_csv:
        writer_train = csv.writer(framelist_train_csv, delimiter=' ', quotechar="'")
        writer_train.writerow(["original_video_id", "video_idx", "frame_idx", "path", "labels"])
        for gt_video_name in train_gt_video_names_unique:
            video_frame_dir_path = os.path.join(cfg.CUSTOM_DATASET.FRAME_DIR, gt_video_name)
            frames = sorted(os.listdir(video_frame_dir_path)) if os.path.exists(video_frame_dir_path) else []
            frame_idx = 0
            for frame in frames:
                relative_frame_path = gt_video_name + "/" + frame
                writer_train.writerow([gt_video_name, video_idx, frame_idx, relative_frame_path, '\"\"'])
                frame_idx = frame_idx + 1
            video_idx = video_idx + 1 if frames else video_idx

    with open(framelist_val_csv_path, "w") as framelist_val_csv:
        writer_val = csv.writer(framelist_val_csv, delimiter=' ', quotechar="'")
        writer_val.writerow(["original_video_id", "video_idx", "frame_idx", "path", "labels"])
        for gt_video_name in valtest_gt_video_names_unique:
            video_frame_dir_path = os.path.join(cfg.CUSTOM_DATASET.FRAME_DIR, gt_video_name)
            frames = sorted(os.listdir(video_frame_dir_path)) if os.path.exists(video_frame_dir_path) else []
            frame_idx = 0
            for frame in frames:
                relative_frame_path = gt_video_name + "/" + frame
                writer_val.writerow([gt_video_name, video_idx, frame_idx, relative_frame_path, '\"\"'])
                frame_idx = frame_idx + 1
            video_idx = video_idx + 1 if frames else video_idx

    logger.info("Created files in framelist dir: " + cfg.CUSTOM_DATASET.FRAME_LIST_DIR)


def create_folder(path_to_new_folder):
    """
    Creates a new folder at path_to_new_folder, if it not already exists
    :param path_to_new_folder: the full path to the to be created folder
    :return:
    """
    if not os.path.exists(path_to_new_folder):
        try:
            os.makedirs(path_to_new_folder)
        except Exception:
            pass


def is_gt_video_file(file_name, gt_video_names_unique):
    """
    Determines wheter an input file is a ground truth (gt) video
    :param file_name: the filename with extension (e.g. video1.avi)
    :return: True, if the file is a gt video
        False, else
    """
    mimestart = mimetypes.guess_type(file_name)[0]

    if mimestart != None:
        mimestart = mimestart.split('/')[0]
        if mimestart == 'video':
            video_name = Path(file_name).stem
            if video_name in gt_video_names_unique:  # is the video referenced in gt file?
                return True

    return False


def compute_mean_and_std(cfg, progress_callback):
    """
    Computes the mean and standard deviation values for the video raw pixels
    across the R G B channels. This is done only on training data and then also
    applied to validation and test
    Calculation is based on https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5
    :param cfg: the config
    :return:
    """

    train_custom_image_dataset = Custom_Image(cfg, "train")

    # A train_image_loader with batch_size 1.
    # This is necessary because different videos have different shapes
    train_image_loader = DataLoader(
        train_custom_image_dataset,
        batch_size=cfg.PREPROCESS.COMPUTE_RGB_MEAN_AND_STD_DATALOADER_BATCH_SIZE,
        num_workers=cfg.PREPROCESS.COMPUTE_RGB_MEAN_AND_STD_DATALOADER_NUM_WORKERS,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.

    # For each images
    for data in train_image_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    # To test, whether all images have been used for computation
    assert nb_samples == train_custom_image_dataset.__len__()

    mean /= nb_samples
    std /= nb_samples

    cfg.DATA.MEAN = mean.tolist()
    cfg.DATA.STD = std.tolist()

    # Export files:
    data_mean_path = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, "data_mean.csv")
    data_std_path = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, "data_std.csv")

    with open(data_mean_path, "w", newline='') as data_mean_csv:
        writer = csv.writer(data_mean_csv, quoting=csv.QUOTE_ALL)
        writer.writerow(cfg.DATA.MEAN)

    with open(data_std_path, "w", newline='') as data_std_csv:
        writer = csv.writer(data_std_csv, quoting=csv.QUOTE_ALL)
        writer.writerow(cfg.DATA.STD)

    logger.info("Computed mean and std of training data over %d images" % nb_samples)
    logger.info("---> Mean for RGB: {}".format(cfg.DATA.MEAN))
    logger.info("---> Std for RGB: {}".format(cfg.DATA.STD))


def compute_test_predict_boxes_and_create_file(cfg, progress_callback=None):
    """
    Creates the CUSTOM_DATASET.TEST_PREDICT_BOX_LISTS file,
    which is used for testing the data
    :param cfg: the config
    :return:
    """
    val_custom_image_dataset = Custom_Image(cfg, "val")

    # The output file
    test_predict_box_list_csv_path = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.TEST_PREDICT_BOX_LISTS[0])

    # use Detectron2 to predict image contents
    object_predictor = PreprocessDetectron2ObjectPredictor(cfg)

    # write the new csv file
    with open(test_predict_box_list_csv_path, "w") as test_predict_box_list_csv:
        writer = csv.writer(test_predict_box_list_csv, quotechar="'")
        # The index of the current video
        video_count = 0
        for video in val_custom_image_dataset._image_paths:
            video_name = val_custom_image_dataset._video_idx_to_name[video_count]
            for middle_frame_timestamp in range(max(cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND, 1),
                                                cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND + 1):

                # Get the frame, for which to make the prediction
                corresponding_middle_frame = sec_to_frame(middle_frame_timestamp, cfg, mode="preprocess")
                # Index starts from 0
                corresponding_middle_frame_idx = corresponding_middle_frame - 1

                if corresponding_middle_frame < len(video):
                    # The path to the frame for which we compute the prediction
                    path_to_middle_frame = video[corresponding_middle_frame_idx]

                    if os.path.isfile(path_to_middle_frame):
                        # Compute prediction
                        pred_boxes, scores = object_predictor.make_prediction_for_preprocess(path_to_middle_frame,
                                                                                             normalize_boxes=True)

                        # Write one line for each prediction
                        for pred_box, score in zip(pred_boxes, scores):
                            # Extract prediction values and round to 3 digits
                            x1_norm = '{:.3f}'.format(round(pred_box.data[0].item(), 3))
                            y1_norm = '{:.3f}'.format(round(pred_box.data[1].item(), 3))
                            x2_norm = '{:.3f}'.format(round(pred_box.data[2].item(), 3))
                            y2_norm = '{:.3f}'.format(round(pred_box.data[3].item(), 3))
                            score = '{:.6f}'.format(round(score.data.item(), 6))
                            # Add line to csv file
                            writer.writerow(
                                [video_name, middle_frame_timestamp, x1_norm, y1_norm, x2_norm, y2_norm, '', score])
                # We reached the end of the video
                else:
                    break
            video_count += 1

        logger.info("Created test file for the test videos at: %s" % test_predict_box_list_csv_path)


def compute_train_predict_box_list_and_create_file(cfg, visualize_results=False, progress_callback=None):
    """
    Creates the CUSTOM_DATASET.TRAIN_GT_BOX_LISTS file, which may server as training input
    only selects detections with iou > 0
    :param cfg: the config
    :param visualize_results: (boolean) whether the iou results should be visualized in an image
        in this image gt-boxes are black and predicted boxes are green
    :return:
    """

    train_dataset = build_dataset("custom", cfg, "train")

    # The output file
    train_predict_box_list_csv_path = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR,
                                              cfg.CUSTOM_DATASET.TRAIN_PREDICT_BOX_LISTS[0])
    logger.info("Creating train_predict_box_list file for the train videos at: %s" % train_predict_box_list_csv_path)

    # use Detectron2 to predict image contents
    object_predictor = PreprocessDetectron2ObjectPredictor(cfg)

    # Used to combine the information from keyframe_indices anCHECK_AND_COPY_FILES_TO_ANNOTATION_DTRAIN_GT_BOX_LISTSIRd keyframe_boxes_and_labels
    key_fame_indices_idx = -1

    # The new csv file
    with open(train_predict_box_list_csv_path, "w") as train_predict_box_list_csv:
        writer = csv.writer(train_predict_box_list_csv, quotechar="'")
        writer.writerow(["video_name", "middle_frame_timestamp", "x1_norm", "y1_norm", "x2_norm", "y2_norm",
                         "action_id", "pred_score", "iou"])

        for video_data, image_paths_for_video, video_idx in zip(train_dataset._keyframe_boxes_and_labels,
                                                                train_dataset._image_paths,
                                                                range(0,
                                                                      len(train_dataset._keyframe_boxes_and_labels))):
            # The name of the current video
            video_name = train_dataset._video_idx_to_name[video_idx]
            for video_second_data, second_idx in zip(video_data, range(0, len(video_data))):
                # Get the corresponding _keyframe_indices information
                key_fame_indices_idx = key_fame_indices_idx + 1

                """
                print("Current index: " + str(key_fame_indices_idx) +
                      " Last index: " + str(len(train_dataset._keyframe_indices)) +
                      " Progress: " +  str(round((key_fame_indices_idx / len(train_dataset._keyframe_indices)),2)))
                """

                # Select the corresponding kreyframe_index_data
                keyframe_index_data = train_dataset._keyframe_indices[key_fame_indices_idx]

                assert (keyframe_index_data[0] == video_idx and keyframe_index_data[1] == second_idx), \
                    "Error in combining the information from keyframe_indices and keyframe_boxes_and_labels"

                # The second for which we will make our prediction and its corresponding frame_number
                middle_frame_timestamp = keyframe_index_data[2]
                frame_number_corresponding_to_middle_frame_timestamp = keyframe_index_data[3]
                # Get the image path (index = frame_number_corresponding_to_middle_frame_timestamp - 1)
                path_to_middle_frame = image_paths_for_video[frame_number_corresponding_to_middle_frame_timestamp - 1]

                # every path has to contain the image name and also
                # "frame_number_corresponding_to_middle_frame_timestamp".file_ending
                assert (video_name in path_to_middle_frame and str(frame_number_corresponding_to_middle_frame_timestamp)
                        + "." in path_to_middle_frame), "The selected path does not match the video or frame_number wrong"

                # Make prediction
                pred_boxes, pred_scores = object_predictor.make_prediction_for_preprocess(path_to_middle_frame,
                                                                                          normalize_boxes=True)

                # If we have predicted boxes
                if pred_scores.nelement() > 0:
                    # tensor with shape (num_predictions, 4 = x1, y1, x2, y2)
                    # Transfer to cpu
                    pred_boxes = pred_boxes.cpu()

                    # gather the gt infos in lists
                    gt_boxes_coords = []
                    gt_boxes_labels = []
                    for box_info in video_second_data:
                        # [x1, y1, x2, y2]
                        gt_boxes_coords.append(box_info[0])
                        # a number
                        gt_boxes_labels.append(box_info[1][0])

                    # compute the iou_tensor with shape (num_preds, num_gt) and convert to list
                    iou_tensor = jaccard(pred_boxes, torch.FloatTensor(gt_boxes_coords))
                    iou_list = iou_tensor.numpy().tolist()

                    # transfer the predictions to cpu and transfrom to list
                    pred_boxes = pred_boxes.numpy().tolist()
                    pred_scores = pred_scores.cpu().numpy().astype(np.float).tolist()

                    for pred_box, pred_score, iou_list_for_pred_box in zip(pred_boxes, pred_scores, iou_list):

                        # Extract prediction values and round to 3 digits
                        x1_norm = '{:.3f}'.format(round(pred_box[0], 3))
                        y1_norm = '{:.3f}'.format(round(pred_box[1], 3))
                        x2_norm = '{:.3f}'.format(round(pred_box[2], 3))
                        y2_norm = '{:.3f}'.format(round(pred_box[3], 3))
                        pred_score = '{:.6f}'.format(round(pred_score, 6))
                        for gt_box_coord, action_id, iou_between_pred_and_gt in zip(gt_boxes_coords, gt_boxes_labels,
                                                                                    iou_list_for_pred_box):
                            """Visualize the results in an image"""
                            if visualize_results:
                                # Read image
                                image = cv2.imread(path_to_middle_frame)

                                # Resoution of image
                                width = image.shape[1]
                                height = image.shape[0]

                                # Draw pred
                                x1 = int(pred_box[0] * width)
                                y1 = int(pred_box[1] * height)
                                x2 = int(pred_box[2] * width)
                                y2 = int(pred_box[3] * height)
                                color_rectangle = (0, 255, 0)  # Green
                                cv2.rectangle(image, (x1, y1), (x2, y2), color_rectangle, 2)

                                # Draw GT
                                x1 = int(gt_box_coord[0] * width)
                                y1 = int(gt_box_coord[1] * height)
                                x2 = int(gt_box_coord[2] * width)
                                y2 = int(gt_box_coord[3] * height)
                                color_rectangle = (0, 0, 0)  # Black
                                cv2.rectangle(image, (x1, y1), (x2, y2), color_rectangle, 2)

                                # Add label
                                label_box = str(round(iou_between_pred_and_gt, 2)) + " |" + str(action_id)
                                t_size = cv2.getTextSize(str(label_box), cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                                cv2.putText(image, str(label_box), (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2,
                                            [0, 0, 0], 2)
                                cv2.imshow("Visualize iou", image)
                                cv2.waitKey(2500)

                            # Use only predictions that have any iou
                            if iou_between_pred_and_gt > 0:
                                iou = '{:.6f}'.format(round(iou_between_pred_and_gt, 6))
                                writer.writerow(
                                    [video_name, middle_frame_timestamp, x1_norm, y1_norm, x2_norm, y2_norm,
                                     action_id, pred_score, iou])
    if visualize_results:
        cv2.destroyAllWindows()

    logger.info("Created train_predict_box_list file for the train videos at: %s" % train_predict_box_list_csv_path)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4=x1,y1,x2,y2]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4=x1,y1,x2,y2]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def sec_to_frame(sec, cfg, mode):
    """
    Convert time index (in second) to frame index.
    [Exemplary values with start second 900, format
    frame_index: time_in_seconds from video start]
    0: 900
    30: 901
    :param sec: a second (int)
    :param cfg: the config
    :param mode: "preprocess" or "demo"
    :return:
    """

    if mode == "preprocess":
        start_second = cfg.PREPROCESS.START_SECOND if cfg.PREPROCESS.START_SECOND > -1 else 0
        return (sec - start_second) * cfg.CUSTOM_DATASET.FRAME_RATE
    elif mode == "demo":
        start_second = 0
        return (sec - start_second) * cfg.CUSTOM_DATASET.FRAME_RATE


def assert_valid_frame_range(cfg):
    """
    Asserts that valid frames are inside the cutted video boundaries
    ranges
    :param cfg:
    :return:
    """

    if (
            cfg.PREPROCESS.START_SECOND > -1 and cfg.PREPROCESS.END_SECOND > -1 and cfg.PREPROCESS.START_SECOND < cfg.PREPROCESS.END_SECOND) \
            or (cfg.PREPROCESS.START_SECOND == -1 and cfg.PREPROCESS.END_SECOND == -1):
        if cfg.PREPROCESS.START_SECOND > -1 and cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND < cfg.PREPROCESS.START_SECOND:
            logger.warning("Attention: VALID_FRAMES_START_SECOND was below start second of cutted video: {}".format(
                cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND))
            cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND = cfg.PREPROCESS.START_SECOND
            logger.warning("Attention: VALID_FRAMES_START_SECOND corrected to start second of cutted video: {}".format(
                cfg.CUSTOM_DATASET.VALID_FRAMES_START_SECOND))
        if cfg.PREPROCESS.END_SECOND > -1 and cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND > cfg.PREPROCESS.END_SECOND:
            logger.warning("Attention: VALID_FRAMES_END_SECOND was after end of cutted video: {}".format(
                cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND))
            cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND = cfg.PREPROCESS.END_SECOND
            logger.warning("Attention: VALID_FRAMES_END_SECOND was corrected end second of cutted video: {}".format(
                cfg.CUSTOM_DATASET.VALID_FRAMES_END_SECOND))
    else:
        logger.warning(
            "Attention: logical error for PREPROCESS.START_SECOND and PREPROCESS.END_SECOND. Please check config")
        sys.exit()

    logger.info("Validity of time range checked")


def preprocess_data(cfg):
    """
    Contains the main logic for preprocessing a custom dataset
    :param cfg:
    :return:
    """
    # Set up environment.
    setup_environment()
    # Setup logging format
    logging.setup_logging(cfg.OUTPUT_DIR)

    logger.info("=== Preprocessing started ===")

    create_folder_structure(cfg)

    assert_valid_frame_range(cfg)

    train_gt_video_names_unique, valtest_gt_video_names_unique = get_unique_gt_video_information(cfg)
    if cfg.PREPROCESS.EXTRACT_FRAMES_FROM_GT_VIDEOS:
        extract_frames_from_gt_videos(cfg, train_gt_video_names_unique + valtest_gt_video_names_unique)

    if cfg.PREPROCESS.CREATE_FRAMELIST_FILES:
        create_framelist_files(cfg, train_gt_video_names_unique, valtest_gt_video_names_unique)

    if cfg.PREPROCESS.COMPUTE_RGB_MEAN_AND_STD:
        compute_mean_and_std(cfg)

    if cfg.PREPROCESS.COMPUTE_TEST_PREDICT_BOXES_AND_CREATE_FILE:
        compute_test_predict_boxes_and_create_file(cfg)

    if cfg.PREPROCESS.COMPUTE_TRAIN_PREDICT_BOX_LIST_AND_CREATE_FILE:
        compute_train_predict_box_list_and_create_file(cfg, visualize_results=False)

    logger.info("=== Preprocessing finished ===")

def extract_frames_from_videos_and_create_framelist_files(cfg, progress_callback=None):
    """
    Called by the gui to extract frames from videos and create framelist files
    :param cfg: the cfg files
    :param progress_callback:
    """
    train_gt_video_names_unique, valtest_gt_video_names_unique = get_unique_gt_video_information(cfg)

    extract_frames_from_gt_videos(cfg, train_gt_video_names_unique + valtest_gt_video_names_unique)

    create_framelist_files(cfg, train_gt_video_names_unique, valtest_gt_video_names_unique)