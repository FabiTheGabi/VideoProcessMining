# Created in the VideoProcessMining project

import os
from pathlib import Path

import numpy as np
import cv2
import multiprocessing as mp
import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment
from slowfast.utils.ava_eval_helper import read_labelmap

POISON_PILL = "STOP"
logger = logging.get_logger(__name__)


class VideoVisualizer(object):
    """
    Used for displaying our demo results
    Displays Object Tracks with Action Labels (action labels are updated every second
    Potentially exports video to self.cfg.DEMO.OUTPUT_FOLDER
    """

    def __init__(self, cfg, img_height, first_middle_frame_index, frames_per_second,
                 input_detection_queue=None, output_detection_queue=None,
                 output_tracker_queue_visualization=None, output_tracker_queue_action_recognition=None,
                 input_action_recognition_queue=None, output_action_recognition_queue_visualization=None,
                 output_action_recognition_queue_result_export=None):
        """
        Initialize the object
        :param cfg: our demo config
        :param img_height: (int) the height of the image
        :param first_middle_frame_index: (int) the index of the first middle_frame index
        :param frames_per_second: (float) the fps of the video -> required for determining middle frames
        :param input_detection_queue: please refer to class MultiProcessDemo
        :param output_detection_queue: please refer to class MultiProcessDemo
        :param output_tracker_queue_visualization: please refer to class MultiProcessDemo
        :param output_tracker_queue_action_recognition: please refer to class MultiProcessDemo
        :param input_action_recognition_queue: please refer to class MultiProcessDemo
        :param output_action_recognition_queue_visualization: please refer to class MultiProcessDemo
        :param output_action_recognition_queue_result_export: please refer to class MultiProcessDemo
        """

        setup_environment()
        # Setup logging format
        logging.setup_logging(cfg.OUTPUT_DIR)

        self.cfg = cfg

        # The name of the input video
        self.demo_video_name = Path(self.cfg.DEMO.VIDEO_SOURCE_PATH).stem

        # Whether we will export an image
        self.export_video = self.cfg.DEMO.VIDEO_EXPORT_VIDEO_ENABLE

        if self.export_video:
            # number of digits for exporting the images (determines how many images can be stored)
            self.number_of_digits_for_image_export = 10
            # The path of the to be created video
            self.export_video_path = os.path.join(self.cfg.DEMO.OUTPUT_FOLDER, self.demo_video_name + "_annotated.mp4")

        # Whether we will display an image
        self.display_video = self.cfg.DEMO.VIDEO_SHOW_VIDEO_ENABLE

        self.cv2_display_name = "Demo: " + self.demo_video_name

        # Whether we will display the meta information (Queues Sizes and img idx)
        self.display_meta_info = cfg.DEMO.VIDEO_SHOW_VIDEO_DEBUGGING_INFO
        # Used for finding the position of meta info
        self.img_height = img_height
        # Used for determining middle_frame_indices (they have the action prediction)
        self.first_middle_frame_index = first_middle_frame_index
        self.frames_per_second = frames_per_second

        # Additional options for displaying the video
        self.video_display_scaling_factor = cfg.DEMO.VIDEO_DISPLAY_SCALING_FACTOR
        self.video_action_display_duration_milliseconds = cfg.DEMO.VIDEO_ACTION_DISPLAY_DURATION_MILLISECONDS

        # The queues containing relevant information
        self.input_detection_queue = input_detection_queue
        self.output_detection_queue = output_detection_queue,
        self.output_tracker_queue_visualization = output_tracker_queue_visualization
        self.output_tracker_queue_action_recognition = output_tracker_queue_action_recognition,
        self.input_action_recognition_queue = input_action_recognition_queue
        self.output_action_recognition_queue_visualization = output_action_recognition_queue_visualization
        self.output_action_recognition_queue_result_export = output_action_recognition_queue_result_export
        # The queue sizes as specified in the config files
        self.queue_size = self.cfg.DEMO.QSIZE_SECONDS * self.cfg.CUSTOM_DATASET.FRAME_RATE

        # Used for terminating the process successfully
        self.action_recognition_input_finished = False

        # The information for displaying actions
        # Load the categories:
        self.path_to_label_map_file = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.LABEL_MAP_FILE) \
            if not os.path.isfile(cfg.ACTIONRECOGNIZER.LABEL_MAP_FILE) \
            else cfg.ACTIONRECOGNIZER.LABEL_MAP_FILE
        # List of dicts (id, name)
        self.action_categories, _ = read_labelmap(self.path_to_label_map_file)
        # A color value for every category
        self.palette_actions = np.random.randint(64, 128, (len(self.action_categories), 3)).tolist()

        # The information required for displaying person_tracking info
        self.palette_person_ids = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

        # The process for person detection
        self.display_next_frame_process = mp.Process(target=self.display_and_or_export_next_frame,
                                                     args=())

        # Used to test the correct order of images
        self.display_img_idx = -1

        # The information for action info display
        self.current_action_output_img_idx = ""
        self.current_pred_action_category_scores = ""

    def start(self):
        """
        Starts the process
        :return:
        """
        self.display_next_frame_process.start()
        print(str(os.getpid()) + ": Video_Visualizer_Process started")

    def join(self):
        """
        Joins the process
        :return:
        """
        self.display_next_frame_process.join()
        print(str(os.getpid()) + "Video_Visualizer_Process joined")

    def display_and_or_export_next_frame(self):
        """
        The logic for adding information to an image and displaying it
        :return:
        """
        print(str(os.getpid()) + ": display_and_or_export_next_frame started")
        while True:
            self.display_img_idx += 1
            # Get the new value from the input queue
            img_idx, img, person_tracking_outputs = self.output_tracker_queue_visualization.get()

            # Break if poison... nothing more to do
            if str(img_idx) == POISON_PILL:
                if self.display_video:
                    cv2.destroyAllWindows()
                if self.export_video:
                    self.create_video_based_on_frames()
                    self.delete_frames_from_output_folder()
                print(str(os.getpid()) + ": display_and_or_export_next_frame break")
                break

            # Assert that img_idxs are ascending without missing idxs
            assert img_idx == self.display_img_idx, "img_idx: " + str(img_idx) + "!= self.display_img_idx: " + \
                                                    str(self.display_img_idx)

            # Add the person tracking information
            img = self.add_person_tracking_info(img, person_tracking_outputs)

            # True, if the img_idx belongs to a middle_frame
            is_middle_frame_image = self.is_middle_frame(img_idx)

            # For a middle_frame we add additional information stemming from action prediction
            img = self.add_action_recognition_info(img, img_idx, person_tracking_outputs, is_middle_frame_image)

            # Display img_idx, queue sizes ect. when showing the images
            if self.display_meta_info:
                img = self.add_meta_info(img_idx, img)

            # display the image
            if self.display_video:
                self.display(img, img_idx, is_middle_frame_image)

            if self.export_video:
                self.save_single_frame(img, img_idx)

    def display(self, img, img_idx, is_middle_frame_image):
        """
        Display an image or not depending on queue size et.c
        :param img: image of shape (H, W, C) (in BGR order) and [0,255])
        :param img_idx: (int) the idx of the image
        :param is_middle_frame_image: whether the image is middle frame (potentially predicted actions)
        """

        # We display every middle frame
        if is_middle_frame_image:
            cv2.imshow(self.cv2_display_name, self.resize_image(img))
            # Pause so that one can see the labels
            cv2.waitKey(self.video_action_display_duration_milliseconds)
        # If the queue is too big, we do not display every image
        elif self.output_tracker_queue_visualization.qsize() > 150:
            # Display only every second image
            if img_idx % 10 == 0:
                cv2.imshow(self.cv2_display_name, self.resize_image(img))
                cv2.waitKey(1)
        elif self.output_tracker_queue_visualization.qsize() > 130:
            # Display only every second image
            if img_idx % 6 == 0:
                cv2.imshow(self.cv2_display_name, self.resize_image(img))
                cv2.waitKey(1)
        elif self.output_tracker_queue_visualization.qsize() > 100:
            # Display only every second image
            if img_idx % 4 == 0:
                cv2.imshow(self.cv2_display_name, self.resize_image(img))
                cv2.waitKey(1)
        # If the queue is too big, we do not display every image
        elif self.output_tracker_queue_visualization.qsize() > 50:
            # Display only every second image
            if img_idx % 3 == 0:
                cv2.imshow(self.cv2_display_name, self.resize_image(img))
                cv2.waitKey(1)
        # If the queue is too big, we do not display every image
        elif self.output_tracker_queue_visualization.qsize() > 10:
            # Display only every second image
            if img_idx % 2 == 0:
                cv2.imshow(self.cv2_display_name, self.resize_image(img))
                cv2.waitKey(1)
        # if the queue is not too big, display the image
        else:
            cv2.imshow(self.cv2_display_name, self.resize_image(img))
            cv2.waitKey(1)

    def save_single_frame(self, img, img_idx):
        """
        Saves a single, annotated frame at self.cfg.DEMO.OUTPUT_FOLDER
        :param img: image of shape (H, W, C) (in BGR order) and [0,255])
        :param img_idx: (int) the idx of the image
        """

        image_export_path = os.path.join(self.cfg.DEMO.OUTPUT_FOLDER,
                                         str(img_idx + 1).zfill(self.number_of_digits_for_image_export) + ".jpg")
        cv2.imwrite(image_export_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def create_video_based_on_frames(self):
        """
        Combines the previously saved images to one video at cfg.CUSTOM_DATASET.FRAME_RATE
        And afterwards deletes the images
        """
        ffmpeg_input_images = self.cfg.DEMO.OUTPUT_FOLDER + "/%0" + str(
            self.number_of_digits_for_image_export) + "d.jpg"

        ffmpeg_command = "ffmpeg -y -loglevel panic -framerate " + str(self.cfg.CUSTOM_DATASET.FRAME_RATE) + " -i " + \
                         ffmpeg_input_images + " -vcodec mpeg4 -b 5000k " + self.export_video_path

        os.system(ffmpeg_command)

        logger.info("Exported Demo Video to " + self.export_video_path)

    def delete_frames_from_output_folder(self):
        """
        Deletes the frames from the output folder, so that only the newly created video remains
        :return:
        """

        # Get all items in the folder
        folder_items = os.listdir(self.cfg.DEMO.OUTPUT_FOLDER)

        # Delete an item, if it is an image
        for folder_item in folder_items:
            if folder_item.endswith(".jpg"):
                os.remove(os.path.join(self.cfg.DEMO.OUTPUT_FOLDER, folder_item))

        logger.info("Cleare output path of all jpg files " + self.export_video_path)

    def resize_image(self, img):
        """
        Resizes an image based on self.video_display_scaling_factor
        :param img: image of shape (H, W, C) (in BGR order) and [0,255])
        :return:
            the resized image of shape shape (H, W, C) (in BGR order) and [0,255])
        """
        if self.video_display_scaling_factor <= 0:
            logger.info("Resizing by factor <= 0 not possible")
            return img
        elif int(self.video_display_scaling_factor) == 1:
            # no changes required
            return img
        else:
            new_width = int(img.shape[1] * self.video_display_scaling_factor)
            new_height = int(img.shape[0] * self.video_display_scaling_factor)
            new_dim = (new_width, new_height)
            resized_image = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
            return resized_image

    def draw_boxes(self, img, bbox, identities=None, offset=(0, 0)):
        """
        Draw the boxes detected in person tracking and also display object ID
        :param img: the image
        :param bbox:
        :param identities:
        :param offset:
        :return:
        """
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_person_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def compute_color_for_person_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette_person_ids]
        return tuple(color)

    def add_person_tracking_info(self, img, person_tracking_outputs):
        """
        Adds the tracking information to an image (bounding box and ID)
        :param img: image of shape (H, W, C) (in BGR order) and [0,255])
        :param person_tracking_outputs: see class MultiProcessDemo
        :return: img: image of shape (H, W, C) (in BGR order) and [0,255]), with bounding box and ID info
        """
        if len(person_tracking_outputs) > 0:
            bbox_xyxy = person_tracking_outputs[:, :4]
            identities = person_tracking_outputs[:, -1]
            img = self.draw_boxes(img, bbox_xyxy, identities)

        return img

    def add_action_recognition_info(self, img, img_idx, person_tracking_outputs, is_middle_frame_image):
        """
        Add the action recognition information to the image, if the action prediction score is
        >= self.cfg.DEMO.VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH
        :param img: ndarray with shape (H, W, C) which is the middle frame image
        :param img_idx: (int) the image id
        :param person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
                                  --> if empty it is a list []
        :param is_middle_frame_image: (bool) whether the current frame is a middle frame image
                if yes, we update the action recognition info, else we do not change it
        :return:
        """
        # We will not be able to receive further data from the output_action_recognition_queue_visualization
        # and thus do not change the image
        if self.action_recognition_input_finished:
            return img
        else:
            if is_middle_frame_image:
                # Get relevant information for display
                self.current_action_output_img_idx, _, self.current_pred_action_category_scores = self.output_action_recognition_queue_visualization.get()

            # The process should terminate by itself in display_and_or_export_next_frame
            if str(self.current_action_output_img_idx) == POISON_PILL:
                # We will not receive any other value from the output_action_recognition_queue_visualization
                self.action_recognition_input_finished = True
                return img

            if is_middle_frame_image:
                assert self.current_action_output_img_idx == img_idx, "Mismatch in image idx"

            if len(self.current_pred_action_category_scores) > 0:
                # only the scores >= tresh. tensor with shape(num_boxes, num_categories)
                display_categories_masks = self.current_pred_action_category_scores >= self.cfg.DEMO.VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH
                # list of arrays with len num_boxes. Each array element=box has only category_ids with scores >= tresh
                display_category_idxs_per_box = [np.nonzero(display_categories_mask)[0]
                                                 for display_categories_mask in display_categories_masks]
                # list of arrays with len num_boxes. Each array element=box has only category_names with scores >= tresh
                display_categories_names_per_box = [
                    [self.action_categories[display_category_id]["name"] for display_category_id in box]
                    for box in display_category_idxs_per_box
                ]

                # Overlay boxes with corresponding categories >= predefined thresh
                for box, category_names, category_idxs, pred_action_category_score in \
                        zip(person_tracking_outputs, display_categories_names_per_box, display_category_idxs_per_box,
                            self.current_pred_action_category_scores):

                    # Add person box (we start from x1, y2) left lower corner of bounding box
                    label_origin = [box[0], box[3]]

                    # Add the labels
                    for category_name, category_idx in zip(category_names, category_idxs):
                        # Get the score for the action category, to adjust width of colour rectangle
                        score = pred_action_category_score[category_idx]
                        category_name = category_name + " (" + str(int(round(score * 100))) + ")"

                        label_origin[-1] -= 5
                        (_, label_height), _ = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
                        # Set label_width to bounding box width
                        label_width = int(round((box[2] - box[0]) * score))
                        cv2.rectangle(
                            img,
                            (label_origin[0], label_origin[1] + 5),
                            (label_origin[0] + label_width, label_origin[1] - label_height - 5),
                            self.palette_actions[category_idx], -1
                        )
                        cv2.putText(
                            img, category_name, tuple(label_origin),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                        )
                        label_origin[-1] -= label_height + 5
            return img

    def add_meta_info(self, img_idx, img):
        """
        Adds the img_idx, and also the queues sizes to the image, which is useful for detecting bottle necks
        :param img_idx: (int) the image_idx
        :param img: image of shape (H, W, C) (in BGR order) and [0,255])
        :return: image of shape (H, W, C) (in BGR order) and [0,255]) with the additional information
        """
        label_idx = "IDX: " + str(img_idx).zfill(7)
        img = cv2.putText(img, label_idx, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        q1 = "(1): " + str(self.input_detection_queue.qsize()) \
             + "=" + \
             str(int((self.input_detection_queue.qsize() / self.queue_size) * 100)) + "%"

        q2 = "  |(2): " + str(self.output_detection_queue[0].qsize()) \
             + "=" + \
             str(int((self.output_detection_queue[0].qsize() / self.queue_size) * 100)) + "%"

        q3 = "  |(3):" + str(self.output_tracker_queue_visualization.qsize()) \
             + "=" + \
             str(int((self.output_tracker_queue_visualization.qsize() / self.queue_size) * 100)) + "%"

        queue_sizes = q1 + q2 + q3
        img = cv2.putText(img, queue_sizes, (10, self.img_height - 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))

        q4 = "(4): " + str(self.output_tracker_queue_action_recognition[0].qsize())\
             + "=" + \
             str(int((self.output_tracker_queue_action_recognition[0].qsize() / self.queue_size) * 100)) + "%"

        q5 = "  |(5): " + str(self.input_action_recognition_queue.qsize()) \
             + "=" + \
             str(int((self.input_action_recognition_queue.qsize() / (self.queue_size*2)) * 100)) + "%"

        q6 = "  |(6):" + str(self.output_action_recognition_queue_visualization.qsize()) \
             + "=" +\
             str( int((self.output_action_recognition_queue_visualization.qsize() / self.queue_size) * 100)) + "%"

        q7 = "  |(7):" + str(self.output_action_recognition_queue_result_export.qsize())

        queue_sizes2 = q4 + q5 + q6 + q7

        img = cv2.putText(img, queue_sizes2, (10, self.img_height - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))

        return img

    def is_middle_frame(self, img_idx):
        """
        Checks whether the current img_idx is a middle_frame_index (= Full second index for which we make an action
        prediction)
        :param img_idx: (int) the current index of an image (indices start from 0)
        :param first_middle_frame_index: (int) the index of the first middle frame in the video
        :param frames_per_second: (float) the fps of the image
        :return:
        """
        # A middle frame_index is always frames_per_second * num_seconds + first_middle_frame_index
        is_mfi = (img_idx - self.first_middle_frame_index) % self.frames_per_second == 0 \
                 and img_idx >= self.first_middle_frame_index
        return is_mfi
