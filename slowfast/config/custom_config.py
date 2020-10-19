#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified in the VideoProcessMining project

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.
    # -----------------------------------------------------------------------------
    # CUSTOM_DATASET:
    # A custom dataset in the ava-style.
    # PREPROCESS helps prepare the input according to the following structure
    # -----------------------------------------------------------------------------

    # Whether we will finetune the model
    _C.TRAIN.FINETUNE = False
    # Whether we will only use the head or also the base layer before the head for fine-tuning
    # Usually, in a first run, you use this option with True
    # Then in a second run, you use this option with false and select final model of the first run for TRAIN.CHECKPOINT_FILE_PATH
    _C.TRAIN.FINETUNE_HEAD_ONLY = False
    # When using another pretrained model to load the weigths and we want to start from epoch 1,
    # we can choose this option. Useful for Finetuning step 2 with FINETUNE_HEAD_ONLY = False
    _C.TRAIN.FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE = False

    # Reduce LR for other layers than head by this factor
    _C.TRAIN.FINETUNE_BASE_LR_REDUCTION_FACTOR = 1.0

    # Unfreezes params that are in this list
    _C.TRAIN.FINETUNE_UNFREEZE_PARAM_LIST = []


    _C.CUSTOM_DATASET = CfgNode()

    # -----------------------------------------------------------------------------
    # The following files are copied based on user input in the respective folders
    # -----------------------------------------------------------------------------

    # Filename of groundtruth file for training, assumed to be in ANNOTATION_DIR
    # Files are used to determine the keyframe_indices for training
    # 8 columns, no headers: video_ID (=video_name), middle_frame_timestamp, x1, y1, x2, y2, action_id, person_id
    _C.CUSTOM_DATASET.TRAIN_GT_BOX_LISTS = ["train_groundtruth.csv"]

    # Filename of groundtruth file for valdation and testing, assumed to be in ANNOTATION_DIR
    # Files are used vor validating and testing in the AVAMeter
    # 8 columns, no headers: video_ID (=video_name), middle_frame_timestamp, x1, y1, x2, y2, action_id, person_id
    _C.CUSTOM_DATASET.GROUNDTRUTH_FILE = "val_groundtruth.csv"

    # Filenames of box list files for test, assumed to be in ANNOTATION_DIR
    # A prepared predicted box file is necessary
    # Files are used to determine the keyframe_indices for testing
    # 8 columns, no headers: video_ID (=video_name), middle_frame_timestamp, x1, y1, x2, y2, action_id, score
    _C.CUSTOM_DATASET.TEST_PREDICT_BOX_LISTS = ["val_predicted_boxes.csv"]

    # Additional filenames of box list files for training.
    # These are an addition to TRAIN_GT_BOX_LISTS, if predicted
    # Note that we assume files which
    # contain predicted boxes will have a suffix "predicted_boxes" in the
    # filename.
    _C.CUSTOM_DATASET.TRAIN_PREDICT_BOX_LISTS = ["train_predict_box_list.csv"]

    # The name of the file to the custom dataset label map, assumed to be in ANNOTATION_DIR
    _C.CUSTOM_DATASET.LABEL_MAP_FILE = "label_map_file.pbtxt"

    # If a person can do two actions at the same time (e.g., standing and talking) or not
    # This is relvant for the export in demo_meter
    _C.CUSTOM_DATASET.MULTIPLE_ACTION_POSSIBLE = False

    # The name of the file to the exclusion timestamps (only for val and test), assumed to be in ANNOTATION_DIR
    # 2 columns, no headers: video_ID (=video_name), middle_frame_timestamp
    _C.CUSTOM_DATASET.EXCLUSION_FILE = "val_excluded_timestamps.csv"

    # -----------------------------------------------------------------------------
    # The files and directories are automatically created and then stored in config
    # -----------------------------------------------------------------------------

    # Directory path of subdirectory for frames.
    _C.CUSTOM_DATASET.FRAME_DIR = ""
    # Directory path of subdirectory for files of frame lists.
    _C.CUSTOM_DATASET.FRAME_LIST_DIR = ("")
    # Directory path of subdirectory for annotation files.
    _C.CUSTOM_DATASET.ANNOTATION_DIR = ("")
    # Directory path of subdirectory for demo.
    _C.CUSTOM_DATASET.DEMO_DIR = ("")

    # Filenames of training samples list files. They are expected
    # to be in the FRAME_LIST_DIR and are automatically created based on TRAIN_GT_BOX_LISTS
    # 5 columns, with headers: original_video_id (=video_name), video_idx (artificialy created & starting from 0), frame_id, path, labels
    # csv file seperated by space
    _C.CUSTOM_DATASET.TRAIN_LISTS = ["train.csv"]
    # Filenames of validation and test samples list files. They are expected
    # to be in the FRAME_LIST_DIR and are automatically created based on GROUNDTRUTH_FILE
    # 5 columns, with headers: original_video_id (=video_name), video_idx (artificialy created & starting from 0), frame_id, path, labels
    # csv file seperated by space
    _C.CUSTOM_DATASET.TEST_LISTS = ["val.csv"]

    # -----------------------------------------------------------------------------
    # Further input and options for training
    # -----------------------------------------------------------------------------

    # Training augmentation parameters
    # Whether to use color augmentation method.
    _C.CUSTOM_DATASET.TRAIN_USE_COLOR_AUGMENTATION = False

    # Whether to only use PCA jitter augmentation when using color augmentation
    # method (otherwise combine with color jitter method).
    _C.CUSTOM_DATASET.TRAIN_PCA_JITTER_ONLY = True

    # Eigenvalues for PCA jittering. Note PCA is RGB based.
    _C.CUSTOM_DATASET.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

    # Eigenvectors for PCA jittering.
    _C.CUSTOM_DATASET.TRAIN_PCA_EIGVEC = [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]

    # -----------------------------------------------------------------------------
    # Further input and options for validation and test
    # -----------------------------------------------------------------------------

    # Whether to do horizontal flipping during test.
    _C.CUSTOM_DATASET.TEST_FORCE_FLIP = False

    # Whether to use full test set for validation split.
    _C.CUSTOM_DATASET.FULL_TEST_ON_VAL = False

    # -----------------------------------------------------------------------------
    # Further options for data processing of single frames
    # -----------------------------------------------------------------------------

    # This option controls the score threshold for the predicted boxes to use for training and test.
    # And also determines the threshold which we use for object detection for preprocessing
    _C.CUSTOM_DATASET.DETECTION_SCORE_THRESH = 0.9

    # If use BGR as the format of input frames.
    _C.CUSTOM_DATASET.BGR = False

    # Backend to process image, includes `pytorch` and `cv2`.
    _C.CUSTOM_DATASET.IMG_PROC_BACKEND = "cv2"

    # Start and end seconds (inclusive, from the beginning of original video) for which predictions are made and evaluated
    # Please refer to the comment of PREPROCESS.START_SECOND to choose sensible values
    _C.CUSTOM_DATASET.VALID_FRAMES_START_SECOND = 0  # >= PREPROCESS.START_SECOND
    _C.CUSTOM_DATASET.VALID_FRAMES_END_SECOND = 100000 # <= PREPROCESS.END_SECOND

    # Frames per second for extracting images from videos
    # depends on the action recognition model and thus should be the same for training and test
    _C.CUSTOM_DATASET.FRAME_RATE = 30

    # -----------------------------------------------------------------------------
    # PREPROCESS options
    # Helps prepare the input according to the following structure
    # -----------------------------------------------------------------------------

    _C.PREPROCESS = CfgNode()

    # Whether preprocessing is required
    _C.PREPROCESS.ENABLE = True

    # Whether the frames should be preprocessed based on original videos
    _C.PREPROCESS.EXTRACT_FRAMES_FROM_GT_VIDEOS = False

    # Whether the framelist-files should be created
    _C.PREPROCESS.CREATE_FRAMELIST_FILES = False

    # Determines whether these values should be computed
    # Makes sense when training a model from scratch
    # This is copmuted only on training data and then also
    # applied to validation and test
    _C.PREPROCESS.COMPUTE_RGB_MEAN_AND_STD = False

    # The batch_size of the dataloader
    _C.PREPROCESS.COMPUTE_RGB_MEAN_AND_STD_DATALOADER_BATCH_SIZE = 1

    # Number of workers for the dataloader
    _C.PREPROCESS.COMPUTE_RGB_MEAN_AND_STD_DATALOADER_NUM_WORKERS = 0

    # Whether to compute test_predict_boxes
    _C.PREPROCESS.COMPUTE_TEST_PREDICT_BOXES_AND_CREATE_FILE = False

    # Whether to compute train_predicted_boxes
    # (style like https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv)
    _C.PREPROCESS.COMPUTE_TRAIN_PREDICT_BOX_LIST_AND_CREATE_FILE = False

    # Directory of "raw"/not-preprocessed videos
    _C.PREPROCESS.ORIGINAL_VIDEO_DIR = ""

    # Start and end second (from beginning of original video, inclusive) according to which the videos are cutted
    # If -1 is specified, the video is not cutted.
    # Be aware that, if possible, the values should be outside of CUSTOM_DATASET_VALID_FRAMES
    # to enable taking these frames for prediction into account:
    #   PREPROCESS.START_SECOND = CUSTOM_DATASET.VALID_FRAMES_START_SECOND - Some_Seconds
    #   PREPROCESS.END_SECOND  = CUSTOM_DATASET.VALID_FRAMES_END_SECOND + Some_Seconds
    _C.PREPROCESS.START_SECOND = -1
    _C.PREPROCESS.END_SECOND = -1

    # -----------------------------------------------------------------------------
    # DEMO:
    # The demo settings for the prototype, it is assumed that the demo is used
    # on a custom dataset
    # -----------------------------------------------------------------------------

    _C.DEMO = CfgNode()

    # Whether demo is required
    _C.DEMO.ENABLE = False

    # Determines the queuesizes and is multplied by the fps
    # (i.e. queue_size = DEMO.QSIZE_SECONDS * CUSTOM_DATASET.FRAME_RATE)
    # only input_action_recognition_queue is bigger than the other queues
    _C.DEMO.QSIZE_SECONDS = 7

    # Visualize results in 1 second intervals
    _C.DEMO.VIDEO_SHOW_VIDEO_ENABLE = True
    # Create the demo as .mp4 video file at OUTPUT_FOLDER
    _C.DEMO.VIDEO_EXPORT_VIDEO_ENABLE = False
    # Whether we will display queue-sizes and image idx in our video
    _C.DEMO.VIDEO_SHOW_VIDEO_DEBUGGING_INFO = False
    # THe demo video displays all categories with
    # this minimum thresh
    _C.DEMO.VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH = 0.0
    # Specify path to the video file for the demo
    _C.DEMO.VIDEO_SOURCE_PATH = ""
    # The path for the demo_video, which is a converted video of VIDEO_SOURCE_PATH at CUSTOM_DATASET.FRAME_RATE
    _C.DEMO.VIDEO_SOURCE_PATH_AT_FPS = ""

    # The time in milliseconds a middle frame with action annotations is displayed, >=1
    _C.DEMO.VIDEO_ACTION_DISPLAY_DURATION_MILLISECONDS = 1

    # If the original video resolution is too big or small for screen, resizing is possible, factor must be > 0
    _C.DEMO.VIDEO_DISPLAY_SCALING_FACTOR = 1.0

    # A folder that is automatically created based on CUSTOM_DATASET.DEMO_DIR to store demo-results
    _C.DEMO.OUTPUT_FOLDER = ""

    # Export detections to csv and xes
    _C.DEMO.EXPORT_EXPORT_RESULTS = True
    # Minimum category prediction score to be exported
    # to csv or xes
    _C.DEMO.EXPORT_MIN_CATEGORY_EXPORT_SCORE = 0.30

    # The deep sort model used for demo is based on https://github.com/ZQPei/deep_sort_pytorch

    # See tracker.py, function def _match(self, detections):
    # Finding track works as follows:DEEPSORT_MAX_IOU_DISTANCE
    """
    - in the deep sort code, there are 3 types of tracks: tentative, confirmed and deleted. For a new object, it is classified as tentative in the first n_init frames
    - only the tracks which are classified as confirmed will use feature similarity, other tracks will use IOU similarity
     for a new object, if it can't be matched using IOU similarity in every frame of the first n_init frames, it will be classified as deleted
    """

    # Whe matching detections to tracks
    # First: confirmed tracks are associated using appearance features.
    # Second: remaining tracks are associated together with unconfirmed tracks using IOU.

    # Deep sort configs
    _C.DEMO.DEEPSORT_REID_CKPT = "./deep_sort/deep/checkpoint/ckpt.t7"
    # The matching threshold. Samples with larger distance are considered an invalid match.
    _C.DEMO.DEEPSORT_MAX_DIST = 0.05
    # Person boxes with confidence score <= DEEPSORT_MIN_CONFIDENCE are filtered out
    _C.DEMO.DEEPSORT_MIN_CONFIDENCE = 0.3
    # ROIs that overlap more than this values are suppressed.
    # I think that if several bboxes overlap too much some are filtered out
    _C.DEMO.DEEPSORT_NMS_MAX_OVERLAP = 0.85
    #  Gating threshold. Associations with cost larger than this value are disregarded.
    _C.DEMO.DEEPSORT_MAX_IOU_DISTANCE = 0.3
    # Maximum number of missed misses before a track is deleted
    _C.DEMO.DEEPSORT_MAX_AGE = 300
    # Number of frames that a track remains in initialization phase
    _C.DEMO.DEEPSORT_N_INIT = 30
    # If not None, fix samples per class to at most this number. Removes
    # the oldest samples when the budget is reached.
    _C.DEMO.DEEPSORT_NN_BUDGET = 1000

    # -----------------------------------------------------------------------------
    # DETECTRON2:
    # -----------------------------------------------------------------------------

    _C.DETECTRON = CfgNode()

    # These are the settings for object detection in the demo
    # It is possible to adjust these values based on a given model id from
    # https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    # The here selected model id is: 137849458
    # Name corresponding to model id
    _C.DETECTRON.DETECTION_MODEL_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # Model corresponding to model id
    _C.DETECTRON.MODEL_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

    # The minimum score threshold [0,1] for predicted detections for demo
    _C.DETECTRON.DEMO_PERSON_SCORE_THRESH = 0.5

    # Boxes that have a smaller pixel height are filtered out & not forwarded to the object tracking module
    _C.DETECTRON.DEMO_MIN_BOX_HEIGHT = -1

    # The batch_size for object prediction
    _C.DETECTRON.DEMO_BATCH_SIZE = 1

    # -----------------------------------------------------------------------------
    # Action Recgonizer:
    # This class is the main entry point for the action recognizer
    # -----------------------------------------------------------------------------

    _C.ACTIONRECOGNIZER = CfgNode()

    # Path to the checkpoint to load the initial weights
    # for the model in the demo file. Has to match to you config
    # You can download them from the "link" in the AVA "model" column
    # https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md
    _C.ACTIONRECOGNIZER.CHECKPOINT_FILE_PATH = ""
    # Checkpoint types include `caffe2` or `pytorch`.
    _C.ACTIONRECOGNIZER.CHECKPOINT_TYPE = "pytorch"
    # If change images to BGR as the format of input frames for action recognition.
    _C.ACTIONRECOGNIZER.BGR = False
    # Backend to process image, includes `pytorch` and `cv2`.
    # cv2 seems to be faster
    _C.ACTIONRECOGNIZER.IMG_PROC_BACKEND = "cv2"
    # Only specify the path to the complete annotation file,
    # if the CUSTOM_DATASET.LABEL_MAP_FILE does not include all classes for your model
    # This is the case for the ava dataset
    _C.ACTIONRECOGNIZER.LABEL_MAP_FILE = ""

    # -----------------------------------------------------------------------------
    # CREPEEVALUATION:
    # Used for doing the crepe_dataset_evaluation and preparing the files
    # -----------------------------------------------------------------------------

    _C.CREPEEVALUATION = CfgNode()

    # This file is used to assign cases to our crepe log based on the gt files, columns:
    # Video,Recipe_type,Frame_start,Frame_end,Object_ID,Case_ID_Artificial,Recepy_Duration,Second_start,Second_End,Video_Name
    _C.CREPEEVALUATION.CASE_OVERVIEW_FILE_PATH = ""

