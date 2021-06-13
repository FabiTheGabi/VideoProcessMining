# Created in the VideoProcessMining project
import os
import sys
import numpy as np
import torch
import multiprocessing as mp

from slowfast.datasets import cv2_transform as cv2_transform
from slowfast.datasets import transform as transform
from slowfast.datasets import utils
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment
import slowfast.utils.checkpoint as cu
from slowfast.models.build import build_model_for_demo

# Used to tell depending processes that they should stop
POISON_PILL = "STOP"

logger = logging.get_logger(__name__)


class ActionRecognizer(object):
    """
    Class for activity recognition, based on an model (e.g.) slowfast
    """

    def __init__(self, cfg, img_height, img_width, model_device, first_middle_frame_index, sample_rate, half_seq_len,
                 current_video_second, input_queue_tracker=None, input_queue_images=None, output_queue=None,
                 output_action_recognition_queue_result_export=None):
        """
        Initialize the ActionRecognizer
        :param cfg: the prototype config
        :param img_height: (int) the height of the images
        :param img_width: (int) the width of the images
        :param model_device: (int) the GPU-ID to which to transfer the model to
        :param first_middle_frame_index: (int) the index of the first middle_frame corresponding to current_video_second
        :param sample_rate: (int) the sample rate
        :param half_seq_len: (int) the half length of a sequence, where each sequence has a defined length and
                        comprises the relevant images for action prediction
        :param current_video_second: (int) the video second, corresponding to the first_middle_frame_index
        :param input_queue_tracker: the queue that provides the person tracking outputs
        :param input_queue_images: the queue that provides the images for action inference (only middle frames)
        :param output_queue: the queue that stores the predicted categories with the corresponding people
        """

        setup_environment()
        # Setup logging format
        logging.setup_logging(cfg.OUTPUT_DIR)

        self.cfg = cfg
        self.show_video = self.cfg.DEMO.VIDEO_SHOW_VIDEO_ENABLE or self.cfg.DEMO.VIDEO_EXPORT_VIDEO_ENABLE

        self.model_device = model_device

        # Build the video model and print model statistics.
        self.activity_prediction_model = build_model_for_demo(self.cfg, self.model_device)
        # Load the pretrained model used for demo
        cu.load_demo_checkpoint(self.cfg, self.activity_prediction_model)
        # Set model to eval mode
        self.activity_prediction_model.eval()

        # Register the queues
        self.output_tracker_queue_action_recognition = input_queue_tracker
        self.input_action_recognition_queue = input_queue_images
        self.output_action_recognition_queue_visualization = output_queue
        self.output_action_recognition_queue_result_export = output_action_recognition_queue_result_export

        # Relevant information for image preprocessing
        self.img_height = img_height
        self.img_width = img_width
        # The short size of our images is scaled to this size
        self.crop_size = cfg.DATA.TEST_CROP_SIZE
        self.data_mean = cfg.DATA.MEAN
        self.data_std = cfg.DATA.STD
        # This is very important. Our images are in BGR format from thread_video_reader
        # Note that Kinetics pre-training uses RGB, which may require changing our
        # BGR images (only for inference) to RGB
        self.use_bgr = cfg.ACTIONRECOGNIZER.BGR

        # The process for person detection
        self.recognize_actions_process = mp.Process(target=self.recognize_actions_multi_processing,
                                                    args=())

        # A list that stores all image_idx data from queue and is sorted by get_idxs (ascending)
        # -> image_idx_from_queue[0] is smallest image_idx
        self.image_idx_from_queue = []
        # A list that stores the corresponding image data also from queue. It is also sorted by the image_idx
        # -> corresponding to image_idx_from_queue
        self.image_data_from_queue = []

        # Stores the relevant image_idx for an action prediction, sorted by image_idx
        self.image_idx_for_prediction = []

        # Stores the relevant image data for an action prediction, sorted by image_idx from image_idx_for_prediction
        self.image_data_for_prediction = []

        ######### All the relevant data for retrieving the process data in the correct form

        # The corresponding frame index to any middle_frame_timestamp of interest
        self.first_middle_frame_index = first_middle_frame_index
        # Used to determine whether an index is a middle frame index for which action recognition is done
        self.current_middle_frame_index = self.first_middle_frame_index

        # Used to test the validity of a to be added image_idx
        self.last_image_idx = -1

        self.sample_rate = sample_rate
        self.half_seq_len = half_seq_len
        self.current_video_second = current_video_second

        # The inference frame indices are sampled around the middle frame as defined for slowfast
        # when using ava_dataset.
        # Here we have indices. index = frame number - 1
        self.inference_frame_indices = list(range(self.current_middle_frame_index + 1 - self.half_seq_len,
                                                  self.current_middle_frame_index + 1 + self.half_seq_len,
                                                  self.sample_rate))

        # The length of our "raw" data list has to be equal to this value
        self.batch_size = len(self.inference_frame_indices)

    def get_person_tracking_outputs_for_idx(self, current_middle_frame_idx):
        """
        Retrieves the person_tracking outputs for a given middle_frame_idx from the
        output_tracker_queue_action_recognition
        :param current_middle_frame_idx: (int) the idx of the middle_frame
        :return:
            pred_person_boxes: ndarray(float32), shapre(num_person_ids, 4=x1,y1,x2,y2)
            person_ids: person_ids ndarray (int), shape (num_person_ids, 1), the person id
            person_tracking_outputs:  ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
                                   --> if empty it is a list []
        """
        values_for_correct_second_retrieved = False
        while not values_for_correct_second_retrieved:
            # Get the the outputs from the feauture action recognition queue
            img_idx, person_tracking_outputs = self.output_tracker_queue_action_recognition.get()

            # This case should not happen at all
            if str(img_idx) == POISON_PILL:
                # ToDo delete assert, if never reached
                assert 3 == 4, "Please look at this problem"
                return [], [], []

            elif img_idx == current_middle_frame_idx:
                if len(person_tracking_outputs) > 0:
                    # Separate detection infos and person ids
                    pred_person_boxes, person_ids = np.array_split(person_tracking_outputs, [4], axis=1)
                    pred_person_boxes = pred_person_boxes.astype(np.float32)
                else:
                    # No person detected, give back empty values
                    pred_person_boxes, person_ids = [], []

                return pred_person_boxes, person_ids, person_tracking_outputs

    def recognize_actions_multi_processing(self):
        """
        Make an action recognition based on the input queues
        """
        print(str(os.getpid()) + ": recognize_actions_multi_processing started")
        while True:
            # Get the relevant image data from the queue
            current_video_second, current_middle_frame_index, img_idx, image = \
                self.input_action_recognition_queue.get()

            # Stop the process
            if str(current_middle_frame_index) == POISON_PILL:
                if self.show_video:
                    self.output_action_recognition_queue_visualization.put((POISON_PILL, POISON_PILL, POISON_PILL))
                self.output_action_recognition_queue_result_export.put((POISON_PILL, POISON_PILL, POISON_PILL))
                print(str(os.getpid()) + ": recognize_actions_multi_processing break")
                break
            # Insert data into our queue_data lists
            else:
                assert img_idx == self.last_image_idx + 1, "Attention, queue got disordered"
                self.last_image_idx = img_idx

                # Append the data to our data storage
                self.image_data_from_queue.append(image)
                self.image_idx_from_queue.append(img_idx)

            # We can make an action prediction, because we have all relevant data
            if img_idx == max(self.inference_frame_indices):
                self.fill_prediction_data()
                self.make_action_prediction_and_insert_into_queues()
                self.empty_prediction_data()

            # Move time window one second forward and delete not necessary values
            if current_middle_frame_index != self.current_middle_frame_index:
                # Move forward one second in time and adjust time dependent data
                self.go_to_next_middle_frame_second(current_middle_frame_index, current_video_second)
                # Delete images that will never be used again
                self.delete_useless_queue_data()

    def fill_prediction_data(self):
        """
        Fills the image_data_for_prediction with the required data for action prediction
        :return:
        """
        for pred_relevant_img_idx in self.inference_frame_indices:
            # Used for localizing the relevant data
            index = self.image_idx_from_queue.index(pred_relevant_img_idx)

            self.image_idx_for_prediction.append(self.image_idx_from_queue[index])
            self.image_data_for_prediction.append(self.image_data_from_queue[index])

        # We should have a full batch of values for inference
        assert len(self.image_data_for_prediction) == self.batch_size, "Batch size not matching"

    def empty_prediction_data(self):
        """
        Empty prediction lists
        """
        self.image_idx_for_prediction = []
        self.image_data_for_prediction = []

    def delete_useless_queue_data(self):
        """
        Removes all "queue" data that is certainly not necessary again for any object prediction
        """
        # This is the first img_idx of any still useful image
        first_imgx_idx_not_to_be_deleted = min(self.inference_frame_indices)

        stop_deleting = False

        while not stop_deleting:
            # We have to have at least one element to delete from
            if self.image_idx_from_queue:
                # Delete the unnecessary data
                if self.image_idx_from_queue[0] < first_imgx_idx_not_to_be_deleted:
                    del self.image_idx_from_queue[0]
                    del self.image_data_from_queue[0]
                # Reached first potentially relevant data, stop deleting
                else:
                    stop_deleting = True
            # Also stop deleting, when there is no data in the list
            else:
                stop_deleting = True

    @torch.no_grad()
    def make_action_prediction_and_insert_into_queues(self):
        """
        Makes the action prediction
        :return:
        """
        # Get the corresponding person_tracking_outputs from the queue
        pred_person_boxes, person_ids, person_tracking_outputs = self.get_person_tracking_outputs_for_idx(
            self.current_middle_frame_index)

        if len(pred_person_boxes) > 0:
            # Person(s) identified --> Do action recognition
            # Preprocess our data, with the current function, we have to wait for pred_person_boxes
            inference_frames, inference_boxes = self.prepare_action_inference_input(self.image_data_for_prediction,
                                                                                    pred_person_boxes)

            # Transfer the data to the current GPU device.
            if isinstance(inference_frames, (list,)):
                for i in range(len(inference_frames)):
                    inference_frames[i] = inference_frames[i].cuda(non_blocking=True, device=self.model_device)
            else:
                inference_frames = inference_frames.cuda(non_blocking=True, device=self.model_device)
            inference_boxes = inference_boxes.cuda(non_blocking=True, device=self.model_device)

            # Compute the predictions.
            pred_action_category_scores = self.activity_prediction_model(inference_frames, inference_boxes)
            # Change tensor to ndarray
            pred_action_category_scores = pred_action_category_scores.cpu().data.numpy()
        else:
            # No person identified --> no action recognition required
            pred_action_category_scores = []

        # Put the output into the queues
        if self.show_video:
            self.output_action_recognition_queue_visualization.put((self.current_middle_frame_index,
                                                                    person_tracking_outputs,
                                                                    pred_action_category_scores))

        self.output_action_recognition_queue_result_export.put((self.current_video_second, person_tracking_outputs,
                                                                pred_action_category_scores))

    def go_to_next_middle_frame_second(self, new_middle_frame_index, new_video_second):
        """
        Increases the time information and recycles image data that has to be reused for next
        action recognition
        :param new_middle_frame_index:
        :param new_video_second:
        :return:
        """

        assert new_video_second == (self.current_video_second + 1), "Provided data is not correct"

        # Update to new_video_second
        self.current_video_second = new_video_second
        # Udpate to new_middle_frame_index
        self.current_middle_frame_index = new_middle_frame_index

        # Update to new_middle_frame_index
        self.inference_frame_indices = list(range(self.current_middle_frame_index + 1 - self.half_seq_len,
                                                  self.current_middle_frame_index + 1 + self.half_seq_len,
                                                  self.sample_rate))

    def prepare_action_inference_input(self, imgs, pred_person_boxes):
        """
        Preprocesses the inputs to feed them to our action prediction model
        The preprocessing of the data is analogous to preprocessing test data in tools/test_net.py
        Before returning, we reformat our variables to be able to directly do inference with our activity_prediction_model
        :param imgs: (list of ndarrays with shape (H, W, C)) (in BGR order) and [0,255])
                            the images that are preprocessed
        :param pred_person_boxes: (ndarray(float32) of shape (num_boxes, 4 =x1, y1, x2, y2)) the predicted person boxes
        :return:
            imgs: (list of tensors with shape (1=number_of_batches, C, num_frames, H, W)) the images used for inference
                        Important: they are usually transferred to RGB, since Kinetics pre-training uses RGB
            pred_person_boxes: (tensor,  shape(num_boxes, 5=BatchIdx, x1, y1, x2, y2)) the boxes for the current clip - not normalized.
        """

        if self.cfg.ACTIONRECOGNIZER.IMG_PROC_BACKEND == "pytorch":
            # Transform images to required format for pytorch backend
            if all(img is not None for img in imgs):
                imgs = torch.as_tensor(np.stack(imgs))

            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and pred_person_boxes.
            imgs, pred_person_boxes = self.images_and_boxes_preprocessing(
                imgs, boxes=pred_person_boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)

        else:
            # Preprocess images and pred_person_boxes
            imgs, pred_person_boxes = self.images_and_boxes_preprocessing_cv2(
                imgs, boxes=pred_person_boxes
            )

        # Change to list. If we have a model with multi input arch, a second pathway is created on the basis of imgs
        # Tensor with shape (C, num_frames, H, W) -> List(s) of tensor with same shape
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        # Reformat the tensors included in the list
        # tensor  shape (C, num_frames, H, W) -> shape (1=number_of_batches, C, num_frames, H, W)
        if isinstance(imgs, (list,)):
            for i in range(len(imgs)):
                imgs[i] = torch.unsqueeze(imgs[i], 0)

        # ndarray shape (num_boxes, 4=x1, y1, x2, y2)) -> tensor shape (num_boxes, 4= x1, y1, x2, y2))
        pred_person_boxes = torch.from_numpy(pred_person_boxes)
        # For each box, we add a the batch_id (in our case always 0)
        # tensor shape (num_boxes, 4= x1, y1, x2, y2)) -> tensor shape (num_boxes, 5= batch_id, x1, y1, x2, y2)))
        pred_person_boxes = torch.cat(
            [torch.full((pred_person_boxes.shape[0], 1), float(0)), pred_person_boxes], axis=1
        )

        return imgs, pred_person_boxes

    def images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (list of ndarrays with len num_frames): the images. Each image
                                    is a ndarray with shape (H, W, C)
            boxes (ndarray): the boxes for the current clip - not normalized. shape (num_boxes, 4 = x1, y1, x2, y2)

        Returns:
            imgs (tensor): list of preprocessed images. shape: (C, num_frames, H, W)
            boxes (ndarray): preprocessed boxes. shape (num_boxes, 4 = x1, y1, x2, y2)
        """

        # Assure that boxes have the right size
        boxes = cv2_transform.clip_boxes_to_image(boxes, self.img_height, self.img_width)

        # `transform.py` is list of np.array. However, for AVA like structure, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        # Short side to test_scale. Non-local and STRG uses 256.
        imgs = [cv2_transform.scale(self.crop_size, img) for img in imgs]
        # Boxes have to be adjusted to new image scale
        boxes = [
            cv2_transform.scale_boxes(
                self.crop_size, boxes[0], self.img_height, self.img_width
            )
        ]

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self.data_mean, dtype=np.float32),
                np.array(self.data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self.use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )

        # If you are interested to see, how the images look like, you can activate this
        # export_image(cfg, imgs.permute(1, 0, 2, 3).data.numpy(), [boxes], "demo", "CHW", True, use_bgr)

        return imgs, boxes

    def images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images. shape (num_frames, C, H, W)
            boxes (ndarray): the boxes for the current clip. shape (num_boxes, 4 = x1, y1, x2, y2)

        Returns:
            imgs (tensor): list of preprocessed images. shape: (num_frames, C, H, W)
            boxes (ndarray): preprocessed boxes. shape (num_boxes, 4 = x1, y1, x2, y2)
        """

        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        # Assure that boxes have the right size
        boxes = transform.clip_boxes_to_image(boxes, self.img_height, self.img_width)

        # Test split
        # Resize short side to crop_size. Non-local and STRG uses 256.
        imgs, boxes = transform.random_short_side_scale_jitter(
            imgs,
            min_size=self.crop_size,
            max_size=self.crop_size,
            boxes=boxes,
        )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self.data_mean, dtype=np.float32),
            np.array(self.data_std, dtype=np.float32),
        )

        if not self.use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        # Assure that boxes have the right size
        boxes = transform.clip_boxes_to_image(
            boxes, self.crop_size, self.crop_size
        )

        # # If you are interested to see, how the images look like, you can activate this
        # Images are in shape: numframes, C, H, W
        # export_image(cfg, imgs.data.numpy(), [boxes], "demo", "CHW", True, use_bgr)
        return imgs, boxes

    def start(self):
        """
        Starts the process
        :return:
        """
        self.recognize_actions_process.start()
        print(str(os.getpid()) + ": Action recognizer started")

    def join(self):
        """
        Joins the process
        :return:
        """
        self.recognize_actions_process.join()
        print(str(os.getpid()) + ": Action recognizer joined")
