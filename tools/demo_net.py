# Created in the VideoProcessMining project

import math
import os
import time

import numpy as np
import torch
from pathlib import Path

from demo_VideoProcessMining.action_recognizer import ActionRecognizer
from demo_VideoProcessMining.deep_sort_tracker import DeepSortTracker
from demo_VideoProcessMining.demo_meter import DemoMeter
from demo_VideoProcessMining.detectron2_object_predictor import DemoDetectron2ObjectPredictor
from demo_VideoProcessMining.file_video_stream import FileVideoStream
from demo_VideoProcessMining.video_visualizer import VideoVisualizer
import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment
from tools.preprocess_net import sec_to_frame, create_folder
import mimetypes
import datetime

import multiprocessing as mp

POISON_PILL = "STOP"

logger = logging.get_logger(__name__)


class MultiProcessDemo:
    def __init__(self, cfg, progress_callback):

        # Set up environment.
        setup_environment()

        # Prepare the input video for best demo results
        cfg.DEMO.VIDEO_SOURCE_PATH_AT_FPS = self.create_demo_video_at_target_framerate(cfg.DEMO.VIDEO_SOURCE_PATH,
                                                                                     cfg.CUSTOM_DATASET.FRAME_RATE)

        self.cfg = cfg

        # An output folder for all demo-related output
        output_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.cfg.DEMO.OUTPUT_FOLDER = os.path.join(self.cfg.CUSTOM_DATASET.DEMO_DIR, output_datetime)
        create_folder(self.cfg.DEMO.OUTPUT_FOLDER)
        logger.info("Created output-folder for demo results at: " + self.cfg.DEMO.OUTPUT_FOLDER)


        # (pyqtSignal) used for signaling back the progress for the GUI
        # We currently take the progress as the percentage of distributed images
        self.progress_callback = progress_callback

        # Used for extracting the data frames from the video file
        self.file_video_stream = FileVideoStream(self.cfg.DEMO.VIDEO_SOURCE_PATH_AT_FPS)
        self.video_file_name = Path(self.cfg.DEMO.VIDEO_SOURCE_PATH).stem

        # Whether we display our results
        self.use_video_visualizer = self.cfg.DEMO.VIDEO_SHOW_VIDEO_ENABLE or self.cfg.DEMO.VIDEO_EXPORT_VIDEO_ENABLE

        # Whether we export our output
        self.export_output = self.cfg.DEMO.EXPORT_EXPORT_RESULTS

        # The fps of the video video source
        self.frames_per_second = self.file_video_stream.frames_per_second
        self.video_length_seconds = self.file_video_stream.video_length_seconds

        # Information on the sampling requirements for the
        # video data
        self.sample_rate = self.cfg.DATA.SAMPLING_RATE
        self.num_frames = self.cfg.DATA.NUM_FRAMES
        self.seq_len = self.sample_rate * self.num_frames
        self.half_seq_len = int(self.seq_len / 2)
        self.half_seq_len_seconds = self.half_seq_len / self.frames_per_second

        # The seconds in the video that are suited for inference
        self.earliest_full_start_second = np.math.ceil(self.half_seq_len_seconds)
        self.final_full_second = math.floor(self.video_length_seconds) - math.ceil(self.half_seq_len_seconds)
        # Set the current_second to start. The current second is the second for which we make the prediction
        self.current_video_second = self.earliest_full_start_second

        # Used for telling the gui the progress of our distribute images function [0, final_full_second] seconds
        self.number_of_relevant_frames = (self.final_full_second + 1) * self.frames_per_second

        # The corresponding frame index to any middle_frame_timestamp of interest
        self.first_middle_frame_index = sec_to_frame(self.earliest_full_start_second, self.cfg, mode="demo") - 1
        # Used to determine whether an index is a middle frame index for which action recognition is done
        self.current_middle_frame_index = self.first_middle_frame_index

        # The inference frame indices are sampled around the middle frame as defined for slowfast
        # when using ava_dataset.
        # Here we have indices. index = frame number - 1
        self.inference_frame_indices = list(range(self.current_middle_frame_index + 1 - self.half_seq_len,
                                                  self.current_middle_frame_index + 1 + self.half_seq_len,
                                                  self.sample_rate))
        # Indicates whether the main process should put the next image in the input_detection_queue
        self.next_image_in_relevant_range = self.current_video_second <= self.final_full_second

        # Multiprocessing configs:
        # How many cpus we have
        self.num_cpu = mp.cpu_count()

        # We have 5 processes in parallel in the simplest case of the demo
        # 1. Main, 2. Object Predictor, 3. Deep Sort Tracker, 4. Video Visualizer, 5. Action Recognizer
        self.num_occupied_processes = 5

        assert self.num_cpu >= self.num_occupied_processes, "You need at least " + str(self.num_occupied_processes) + " cores for the multiprocessing demo"

        self.free_cpu_cores = self.num_cpu - self.num_occupied_processes
        # How many gpus we have for the demo
        self.num_gpu = self.cfg.NUM_GPUS

        # How many gpus should be used for object detection (increasing number)
        self.num_gpu_object_detection = min(self.free_cpu_cores, self.num_gpu)

        # The gpuid for action recognition (decreasing or in our case last gpuid
        # We take the las possible gpuid for action recognition because this is beneficiary, if we have
        # less processes than free_cpu_cores (object detection and action recognition are separated this way)
        self.gpuid_action_recognition = self.num_gpu - 1

        # The queue sizes as specified in the config files
        self.queue_size = self.cfg.DEMO.QSIZE_SECONDS * self.cfg.CUSTOM_DATASET.FRAME_RATE

        # Queues
        # Contains the original images with an idx each:
        #   1. img_idx (int)
        #   2. image of shape (H, W, C) (in BGR order) and [0,255])
        self.input_detection_queue = mp.Queue(maxsize=self.queue_size)
        # Queue containing the detections per image in form
        #   1. img_idx (int),
        #   2. image of shape (H, W, C) (in BGR order) and [0,255]),
        #   3. predictions {dict}: a dict with the following keys
        #       pred_boxes: tensor of shape num_predictions, 4 =
        #                   the coordinates of the predicted boxes [x1, y1, x2, y2]) --> if empty it is []
        #       scores: tensor of shape (num_predictions) containing the confidence scores [0,1]) --> if empty it is []
        self.output_detection_queue = mp.Queue(maxsize=self.queue_size)
        # Contains the images with the corresponding ids and person_tracking_outputs -> used for visualization
        #   1. img_idx (int)
        #   2. image of shape (H, W, C) (in BGR order) and [0,255])
        #   3. person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
        #                          --> if empty it is a list []
        self.output_tracker_queue_visualization = mp.Queue(maxsize=self.queue_size)

        # Contains the images with the corresponding ids and person_tracking_outputs -> used for action recognition
        #   1. img_idx (int)
        #   2. person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
        #                          --> if empty it is a list []
        self.output_tracker_queue_action_recognition = mp.Queue(maxsize=self.queue_size)

        # Contains the input for action_recognition (only for img_idxs that are middle_frames)
        #   1. current_video_second: (int) the current video second for which the prediction data is given
        #   2. img_idxs=current_middle_frame_index (int) the image img_idx, which is always the next middle_frame_index
        #   3. img_idx (int) = the idx of the current middle_frame
        #   4. image of shape (H, W, C) (in BGR order) and [0,255])
        # It is bigger than the other queues
        self.input_action_recognition_queue = mp.Queue(maxsize=int(self.queue_size*1.5))

        # Contains the input for action_recognition (only for img_idxs that are middle_frames)
        #   1. img_idx (int), only for middle frames
        #   2. person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
        #                          --> if empty it is a list []
        #   3. pred_action_category_scores (ndarray float32) shape(num_person_ids, num_categories),
        #                                       the scores for each person and each action category
        #                                   --> if empty it is a list []
        self.output_action_recognition_queue_visualization = mp.Queue(maxsize=self.queue_size)

        # Contains the input for action_recognition (only for img_idxs that are middle_frames)
        #   1. current_video_second: (int) the current video second for which the prediction data is given
        #   2. person_tracking_outputs: ndarray with shape (num_identities, 5(int)= x1,y1,x2,y2,identity_number)
        #                          --> if empty it is a list []
        #   3. pred_action_category_scores (ndarray float32) shape(num_person_ids, num_categories),
        #                                       the scores for each person and each action category
        #                                   --> if empty it is a list []
        self.output_action_recognition_queue_result_export = mp.Queue(maxsize=int(self.video_length_seconds * self.frames_per_second))

        # A list of dicts that contains detected middle_frame_seconds
        self.middle_frame_seconds = []

        # The detectron2_object_predictor_class for person detection
        self.object_predictor = DemoDetectron2ObjectPredictor(self.cfg, self.file_video_stream.height,
                                                              self.file_video_stream.width,
                                                              parallel=True,
                                                              num_gpu=self.num_gpu_object_detection,
                                                              input_queue=self.input_detection_queue,
                                                              output_queue=self.output_detection_queue,
                                                              gpuid_action_recognition=self.gpuid_action_recognition)

        # The deep sort tracker class for person tracking
        self.deep_sort_tracker = DeepSortTracker(self.cfg, input_queue=self.output_detection_queue,
                                                 output_queue_vis=self.output_tracker_queue_visualization,
                                                 output_queue_action_pred=self.output_tracker_queue_action_recognition,
                                                 show_video=self.use_video_visualizer)

        # The action recognition class
        self.action_recognizer = ActionRecognizer(self.cfg, self.file_video_stream.height, self.file_video_stream.width,
                                                  model_device=self.gpuid_action_recognition,
                                                  first_middle_frame_index=self.first_middle_frame_index,
                                                  sample_rate=self.sample_rate,
                                                  half_seq_len=self.half_seq_len,
                                                  current_video_second=self.current_video_second,
                                                  input_queue_tracker=self.output_tracker_queue_action_recognition,
                                                  input_queue_images=self.input_action_recognition_queue,
                                                  output_queue=self.output_action_recognition_queue_visualization,
                                                  output_action_recognition_queue_result_export=
                                                  self.output_action_recognition_queue_result_export)

        if self.export_output:
            # Our demo meter to store and finally print the results
            self.demo_meter = DemoMeter(self.cfg, self.file_video_stream.height, self.file_video_stream.width)
            # Used to control the completeness of our export
            self.current_export_second = self.earliest_full_start_second - 1

        if self.use_video_visualizer:
            self.demo_visualizer = VideoVisualizer(self.cfg, self.file_video_stream.height, self.first_middle_frame_index,
                                                   self.frames_per_second,
                                                   input_detection_queue=self.input_detection_queue,
                                                   output_detection_queue=self.output_detection_queue,
                                                   output_tracker_queue_visualization=self.output_tracker_queue_visualization,
                                                   output_tracker_queue_action_recognition=self.output_tracker_queue_action_recognition,
                                                   input_action_recognition_queue=self.input_action_recognition_queue,
                                                   output_action_recognition_queue_visualization=self.output_action_recognition_queue_visualization,
                                                   output_action_recognition_queue_result_export=self.output_action_recognition_queue_result_export)

    def run_demo(self):

        self.start_processes()
        t0 = time.time()
        self.distribute_images()
        # Export our results, before joining the processes
        if self.export_output:
            self.export_results()

        # Test Duration for one run
        duration = time.time() - t0

        self.join_processes()

        # Delete the video created for demo
        self.delete_demo_video(self.cfg.DEMO.VIDEO_SOURCE_PATH_AT_FPS)

        print("Duration: " + str(duration))
        print("Speed Up " + str(self.video_length_seconds / duration))

    def export_results(self):
        """
        Fills the Meter with our action detections and exports results
        """
        poisoned = False
        while not poisoned:
            self.current_export_second += 1

            current_video_second, person_tracking_outputs, pred_action_category_scores = \
                self.output_action_recognition_queue_result_export.get()

            if str(current_video_second) == POISON_PILL:
                poisoned = True
            else:
                assert current_video_second == self.current_export_second, "Export Results: a second is missing"
                # Save the predictions
                self.demo_meter.add_detection(video_id=self.video_file_name,
                                              video_second=current_video_second,
                                              person_tracking_outputs=person_tracking_outputs,
                                              pred_action_category_scores=pred_action_category_scores)
        self.demo_meter.export_results()

    def distribute_images(self):
        """
        This is the heart of the demo's logic, which puts images into the queues and
        orchestrates the complete logic of image processing
        """
        while self.file_video_stream.more() and self.next_image_in_relevant_range:

            # grab the frame from the threaded video file stream
            img_idx, current_frame = self.file_video_stream.read()

            if img_idx == max(self.inference_frame_indices):

                # Prepare necessary data for the next video second
                self.current_video_second += 1
                self.current_middle_frame_index = sec_to_frame(self.current_video_second, self.cfg, mode="demo") - 1
                self.inference_frame_indices = list(range(self.current_middle_frame_index + 1 - self.half_seq_len,
                                                          self.current_middle_frame_index + 1 + self.half_seq_len,
                                                          self.sample_rate))

                # Determines whether we the images of the current second are relevant for prediction
                self.next_image_in_relevant_range = self.current_video_second <= self.final_full_second

            if self.next_image_in_relevant_range:
                # Put image in input_detection_queue
                self.input_detection_queue.put((img_idx, current_frame))
                # Put the data into the input_action_recognition_queue
                self.input_action_recognition_queue.put((self.current_video_second, self.current_middle_frame_index,
                                                         img_idx, current_frame))
                # Update progress bar
                self.progress_callback.emit((img_idx+1)/self.number_of_relevant_frames)

        # This is necessary because we do everything in the main thread, otherwise, the process will not finish
        self.input_action_recognition_queue.put((POISON_PILL, POISON_PILL, POISON_PILL, POISON_PILL))
        print("POISON input_action_recognition_queue from main")
        # Also put poison pills into object predictor
        self.object_predictor.shutdown()

    def create_demo_video_at_target_framerate(self, data_video_source_path, target_framerate):
        """
        Converts the video_source video so that it fits the target fps
        The newly created video name ends with "_demo_{fps}fps".
        Furthermore, the config
        :param data_video_source_path: (string) the path to the video file which is updated to target fps
        :param target_framerate: (int) the new target frame rate
        :return:
            data_video_source_path_new: (string) the path to the new video with the target fps
        """
        assert os.path.isfile(data_video_source_path), "Attention" + data_video_source_path + "contains no file"
        path_to_video = os.path.dirname(data_video_source_path)
        file_name_wo_extension = Path(data_video_source_path).stem
        _, file_extension = os.path.splitext(data_video_source_path)

        # ffmpeg has problems using .webm as output
        if file_extension == ".webm":
            file_extension = ".mp4"

        new_video_path = os.path.join(path_to_video, file_name_wo_extension + "_demo_" +
                                      str(target_framerate) +"fps" + file_extension)


        assert self.is_video(data_video_source_path), "Attention" + data_video_source_path + "is no video file"

        if not os.path.isfile(new_video_path):
            logger.info("Creating new video file with target fps at: " + new_video_path)
            # We use the -y flag to guarantee that an already existing file is overwritten
            ffmpeg_command = "ffmpeg -y -loglevel panic -i " + data_video_source_path + \
                             " -filter:v fps=fps="+ str(target_framerate) +" -q:v 1 " + new_video_path
            os.system(ffmpeg_command)
            logger.info("Created new video file with target fps at: " + new_video_path)
        else:
            logger.info("File already existed at: " + new_video_path)

        return new_video_path

    def delete_demo_video(self, demo_video_source_path):
        """
        Deletes the demo_video not to waste memory
        :param demo_video_source_path: (string) path to the previously created demo video file
        """
        os.remove(demo_video_source_path)
        logger.info("Deleted file at: " + demo_video_source_path)

    def is_video(self, data_video_source_path):
        """
        Determines whether a file is a video file
        :param data_video_source_path: (string) the full path to a video_file
        :return:
            True: if the file is a video file
            False: if not
        """
        # Get the file name (e.g. video1.avi)
        mimestart = mimetypes.guess_type(data_video_source_path)[0]

        if mimestart != None:
            mimestart = mimestart.split('/')[0]
            if mimestart == 'video':
                return True
            else:
                return False

    def start_processes(self):
        """
        Starts the processes
        :return:
        """
        # Start extracting frames from video
        self.file_video_stream.start()
        # Start the processes
        self.object_predictor.start()
        self.deep_sort_tracker.start()
        # Currently not used because not working & not required
        self.action_recognizer.start()
        if self.use_video_visualizer:
            self.demo_visualizer.start()

    def join_processes(self):
        """
        Stops and joins the processes. Puts POISON_PILL into self.output_detection_queue,
        which leads to the termination of dependent processes
        """
        # Only puts POISON_PILL into subsequent queues, but does not terminate and join the child processes
        self.object_predictor.join()
        self.deep_sort_tracker.join()
        self.action_recognizer.join()

        # stop the video stream and join threads
        if self.use_video_visualizer:
            self.demo_visualizer.join()

        # stop the video stream and join threads
        self.file_video_stream.join()


def run_demo(cfg, progress_callback=None):
    """
    :param cfg:
    :return:
    """
    # Set up environment.
    setup_environment()
    logger.info("=== Demo started ===")
    multi_process_demo = MultiProcessDemo(cfg, progress_callback)
    multi_process_demo.run_demo()
    logger.info("=== Demo finished ===")