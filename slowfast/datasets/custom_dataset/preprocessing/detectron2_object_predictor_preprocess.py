# Created in the VideoProcessMining project based on https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

# import some common libraries
import os
import logging

import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

logger = logging.getLogger(__name__)

class PreprocessDetectron2ObjectPredictor(object):
    def __init__(self, cfg):
        """
        Creates an Detectron2 based prediction class
        :param cfg: the config file
        """
        self.cfg = cfg
        self.create_object_predictor(cfg)


    def create_object_predictor(self, mode):
        """
        Creates a dectectron2 DefaultPredictor to run inference on images
        :return: 
        """

        # Get the values from the config file
        detectron2_cfg_file = self.cfg.DETECTRON.DETECTION_MODEL_CFG
        detectron2_model_weights = self.cfg.DETECTRON.MODEL_WEIGHTS
        detectron2_score_tresh_test = self.cfg.CUSTOM_DATASET.DETECTION_SCORE_THRESH

        # Create dectectron2 DefaultPredictor to make person box prediction on images
        detectron2_cfg = get_cfg()
        detectron2_cfg.merge_from_file(model_zoo.get_config_file(detectron2_cfg_file))
        detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detectron2_score_tresh_test
        detectron2_cfg.MODEL.WEIGHTS = detectron2_model_weights

        self.predictor = DefaultPredictor(detectron2_cfg)


    def make_prediction_for_preprocess(self, path_to_image, normalize_boxes):
        """
        Makes a prediction only for the person class for one image
        :param cfg: the config
        :param path_to_image: a path to the image to make the prediction for
        :param normalize_boxes: whether to scale the box_coordinates relative to image size
        :return: pred_boxes: cuda-tensor of shape
                            (num_predictions, 4 = the coordinates of the predicted boxes [x1, y1, x2, y2])
                scores: cuda-tensor of shape (num_predictions), prediction scores [0,1]
        """

        if os.path.isfile(path_to_image):
            # Read image, image of shape (H, W, C) (in BGR order) and [0,255]
            img = cv2.imread(path_to_image)
            # Make prediction
            predictions = self.predictor(img)

            # Extract results
            fields = predictions["instances"]._fields
            pred_classes = fields["pred_classes"]

            # Select only boxes and scores that for persons
            selection_mask = pred_classes == 0
            pred_boxes = fields["pred_boxes"].tensor[selection_mask]
            scores = fields["scores"][selection_mask]

            if scores.nelement() > 0:
                if normalize_boxes:
                    pred_boxes = self.normalize_boxes(img, pred_boxes)

        return pred_boxes, scores

    def normalize_boxes(self, img, boxes):
        """
        Normalizes an image with respect to its frame_size
        :param img: (ndarray) image in format (H, W, C)
        :param boxes: (tensor), shape (num_boxes, 4 = x1, y1, x2, y2) the boxes to be normalized
        :return: (tensor), shape (num_boxes, 4 = x1, y1, x2, y2) the normalized boxes
        """
        image_height = img.shape[0]
        image_width = img.shape[1]
        boxes[:, [0, 2]] /= image_width
        boxes[:, [1, 3]] /= image_height

        return boxes
