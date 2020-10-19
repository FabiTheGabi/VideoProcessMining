# Created in the VideoProcessMining project
import logging
import os
import cv2
import torch
import numpy as np
import bisect
import multiprocessing as mp
import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment

from deep_sort import build_tracker

logger = logging.get_logger(__name__)
POISON_PILL = "STOP"


class DeepSortTracker(object):
    """
    A Deep Sort tracker, fully based on parts of
    https://github.com/ZQPei/deep_sort_pytorch.
    Receives images and person detections and returns the detected person boxes
    """

    def __init__(self, cfg, input_queue=None, output_queue_vis=None, output_queue_action_pred=None,
                 use_gpu=True, show_video=False):
        """
        :param cfg: the prototype config
        :param input_queue: inut
        :param: output_queue: output
        :param use_gpu: (boolean) whether gpu should be use for inference
        """

        setup_environment()

        self.cfg = cfg
        self.show_video = show_video

        # The queues from the main thread used for multiprocessing
        self.output_detection_queue = input_queue
        if self.show_video:
            self.output_tracker_queue_visualization = output_queue_vis
        self.output_tracker_queue_action_recognition = output_queue_action_pred

        # Used to order images retrieved from the two queues used as  input
        self.get_idx = -1

        # Has the previous process terminated?
        self.first_poison_pill_received = False
        # True if self.first_poison_pill_received and get does not lead to any results and self.result_rank is empty
        self.output_detection_queue_is_finished = False

        # Whether we will use a gpu or not
        use_gpu = use_gpu and torch.cuda.is_available()

        # The tracker we will use for object detection
        self.deepsort = build_tracker(cfg, use_cuda=use_gpu)

        # A list that contains and is sorted by get_idxs (ascending) -> result_rank[0] is smallest get_idx
        self.result_rank = []
        # A list that contains the images (ndarray) image with shape (H, W, C) (in BGR order) and [0,255])
        self.result_img_data = []
        # A list that contains the prediction results (predictions {dict}) and that is
        # also sorted by the get_idxs -> corresponding to result_rank
        self.result_prediction_data = []

        # The process for person detection
        self.update_tracker_with_next_image_prcocess = mp.Process(target=self.update_tracker_with_next_image,
                                                                  args=())

    def update_tracker_with_next_image(self):
        """
        Selects the next img_idx from the input queue. This is important
        when there are several processes for object detection, and the correct order is not certainly guaranteed.
        """
        print(str(os.getpid()) + ": update_tracker_with_next_image started")
        while True:
            correct_entry_retrieved = False
            self.get_idx += 1  # the index needed for this request

            # 1. The prediction is already stored in result_date --> return result data
            # If the predictions for the current get_idx have already been extracted from the result_queue
            # stored in result_data, retrieve them and delete the data corresponding to
            # get_idx from the result_rank, result_img_data and result_prediction_data
            # since get_idx is ordered ascending, it is sufficient to check only the first element
            if len(self.result_rank) and self.result_rank[0] == self.get_idx:
                img_idx = self.get_idx
                img = self.result_img_data[0]
                predictions = self.result_prediction_data[0]
                del self.result_rank[0], self.result_img_data[0], self.result_prediction_data[0]

            else:
                # 2. The prediction has to be retrieved from the output_detection_queue
                # since it is not ordered, we have to make sure to get the right idx
                while not correct_entry_retrieved:
                    # remove and return the oldest item (fifo - not necessarily ordered idx) from queue
                    if not self.first_poison_pill_received:
                        img_idx, img, predictions = self.output_detection_queue.get()
                    # We already received a POISON_PILL, but it is possible that there are still valid
                    # items in the queue
                    else:
                        # Try to get the next item
                        try:
                            img_idx, img, predictions = self.output_detection_queue.get(timeout=2)
                        # When we receive no more items, we assume that we have gathered valid items from the queue
                        # we also have processed everything relevant and can return and put POISON Pills into subsequent
                        # queues
                        except Exception:
                            if self.show_video:
                                self.output_tracker_queue_visualization.put((POISON_PILL, POISON_PILL, POISON_PILL))
                            self.output_tracker_queue_action_recognition.put((POISON_PILL, POISON_PILL))
                            print(str(os.getpid()) + ": update_tracker_with_next_image break/return")
                            return  # break would only exit one loop and thus not terminate correctly

                    # If POISON_PILL, only if get does not result into any more values
                    if str(img_idx) == POISON_PILL:
                        self.first_poison_pill_received = True

                    # if the retrieved idx is not equal to the get_idx, we store results for later use
                    elif img_idx != self.get_idx:
                        # if not we have the cae idx > get_idx --> we store the prediction for later use

                        # the index for inserting the element (starting from 0 as first position to insert)
                        # bisect.bisect locates the insertion point in result_rank to keep a sorted order
                        # defined by the get_idx. Thus, the element with the smallest get_idx is always at
                        # the [0] position
                        insert_index = bisect.bisect(self.result_rank, img_idx)
                        self.result_rank.insert(insert_index, img_idx)
                        self.result_img_data.insert(insert_index, img)
                        self.result_prediction_data.insert(insert_index, predictions)
                    else:
                        # Correct entry retreived
                        correct_entry_retrieved = True

            assert img_idx == self.get_idx, "The order of person tracking is wrong"
            # Do necessary computation with the input to the correct get_idx
            bbox_xyxy = predictions.get("pred_boxes")
            scores = predictions.get("scores")
            person_tracking_outputs = self.update_tracker(bbox_xyxy, scores, img)

            # Put results into result queue
            if self.show_video:
                self.output_tracker_queue_visualization.put((img_idx, img, person_tracking_outputs))
            self.output_tracker_queue_action_recognition.put((img_idx, person_tracking_outputs))

    def update_tracker(self, bbox_xyxy, scores, img):
        """
        Updates the tracker with the next image
        :param bbox_xyxy: (ndarray) shape (num_boxes, 4 = x1, y1, x2, y2, not normalized), if empty it is []
        :param scores: (ndarray) shape (num_boxes), the confidence scores, if empty it is []
        :param img: (ndarray) image with shape (H, W, C) (in BGR order) and [0,255]) #ToDo: if batch is required, processing of list items
        :return:
            person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
                         if there are no person_tracking_outputs [] list is returned
        """
        # Change to the required output format
        bbox_xywh = xyxy_to_xywh(bbox_xyxy)
        # We have to convert the image to RGB, since this is used in the original implementation
        inference_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if there are no person_tracking_outputs [] is returned
        # if there are person_tracking_outputs, we have a ndarray with shape (num_identities, 5= x1,y1,x2,y2,identity_number)
        person_tracking_outputs = self.deepsort.update(bbox_xywh, scores, inference_image)

        return person_tracking_outputs

    def start(self):
        """
        Starts the process
        :return:
        """
        self.update_tracker_with_next_image_prcocess.start()
        print(str(os.getpid()) + ": Deep Sort Tracker started")

    def join(self):
        """
        Joins the process
        :return:
        """
        self.update_tracker_with_next_image_prcocess.join()
        print(str(os.getpid()) + ": Deep Sort Tracker joined")


def xyxy_to_xywh(boxes_xyxy):
    """
    Changes bounding box values from xyxy to xywh
    :param boxes_xyxy: (ndarray) shape (num_boxes, 4 = x1, y1, x2, y2, not normalized)
                        tensor would also be possible
    :return:
        boxes_xywh: the boxes in the xywh format
    """
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh
