# Created in the VideoProcessMining project based on https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py and https://github.com/facebookresearch/detectron2/blob/7f713b7a224fbb6d7ee10fe7e7dcd7ed1edfb885/detectron2/engine/defaults.py#L160

# import some common libraries
import os
import time
import torch
import cv2

import multiprocessing as mp
import bisect
from collections import deque

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog

import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment
from slowfast.datasets import cv2_transform
# Used to tell depending processes that they should stop
POISON_PILL = "STOP"

logger = logging.get_logger(__name__)


class DemoDetectron2ObjectPredictor(object):
    def __init__(self, cfg, img_height, img_width, parallel=False, num_gpu=None, input_queue=None, output_queue=None,
                 gpuid_action_recognition=None):
        """
        Creates an Detectron2 based prediction class
        which is optimized for demo and should be used for it.
        The code is slightly modified from the original detectron2 demo content
        :param cfg: the config file for the prototype
        :param img_height: (int) the height of the input images
        :param img_width: (int) the width of input images
        :param parallel: (boolean) whether, we will do asynchronous computation
        :param num_gpu: (int) number of gpus we will use for asynchronous computation
        :param input_queue: (multiprocessing.queue) containing the input images
                            (img_idx, image of shape (H, W, C) (in BGR order) and [0,255])
        :param output_queue: (multiprocessing.queue) containing the computed predictions
        :param gpuid_action_recognition: (int) the gpuid for object tracking

        """

        setup_environment()
        # Setup logging format
        logging.setup_logging(cfg.OUTPUT_DIR)

        # The cfg file for the prototype
        self.cfg = cfg

        # The original image resolution: used for resizing provided images
        self.img_height = img_height
        self.img_width = img_width

        # We only use the demo config
        self.detectron2_cfg_file = self.cfg.DETECTRON.DETECTION_MODEL_CFG
        self.detectron2_model_weights = self.cfg.DETECTRON.MODEL_WEIGHTS
        self.detectron2_score_tresh_test = self.cfg.DETECTRON.DEMO_PERSON_SCORE_THRESH

        # Load the detectron config
        self.detectron_config = self.setup_detectron_config()

        # Can be useful for displaying the object classes
        self.metadata = MetadataCatalog.get(
            self.detectron_config.DATASETS.TEST[0] if len(self.detectron_config.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")

        # Determines whether we will use async processing
        self.parallel = parallel
        if self.parallel:
            # Used for async processing
            self.predictor = AsyncPredictor(self.cfg, self.detectron_config, self.img_height, self.img_width,
                                            num_gpus=num_gpu, input_queue=input_queue, output_queue=output_queue,
                                            gpuid_action_recognition=gpuid_action_recognition)
            # Used to count the frames provided for detect_persons
            self.provided_image_count = 0
            self.buffer_size = self.predictor.default_buffer_size
            # In the original version this attribute was used to store
            # the images in chronological order as well as a counter that represents the size of the task_queue
            # attribute. Since we do not return the images, we only use it as a counter representing the task_queue and
            # thus insert a dummy int variable instead of an image, because it is more memory efficient
            self.frame_data = deque()
        else:
            # Use the modified predictor for the demo
            self.predictor = DemoDefaultPredictor(self.cfg, self.detectron_config, self.img_height, self.img_width)

    def setup_detectron_config(self):
        """
        Sets up the config for our predictors.
        This config is used to build our predictor class
        :return:
            detectron_config: the config for our detectron2 model
        """
        # Get Config
        detectron_config = get_cfg()
        detectron_config.merge_from_file(model_zoo.get_config_file(self.detectron2_cfg_file))
        # We only use the weights
        detectron_config.merge_from_list(["MODEL.WEIGHTS", self.detectron2_model_weights])
        # Set score_threshold for builtin models
        detectron_config.MODEL.RETINANET.SCORE_THRESH_TEST = self.detectron2_score_tresh_test
        detectron_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detectron2_score_tresh_test
        detectron_config.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.detectron2_score_tresh_test
        # Freeze Model
        detectron_config.freeze()

        return detectron_config

    def detect_persons(self, current_frame, is_last_frame):
        """
        :param current_frame (ndarray): the image with shape (H, W, C) in BGR format
        :param is_last_frame: whether this is the las frame provided for prediction
                            Necessary for emptying the buffer
        :return: predictions (list[dict]): a list with an entry for each predicted image (see filter_person_boxes)
        """

        # Make predictions asynchronously
        if self.parallel:
            # Counts the number of provided images
            self.provided_image_count += 1

            # Async prediction
            # Here we only add 1 as a dummy variable instead of current_frame as in original code
            self.frame_data.append(1)
            self.predictor.put(current_frame)

            # After we filled the buffer, we make async predictions
            if self.provided_image_count >= self.buffer_size:
                # Remove an entry to account for predicting one element form the task_queue
                self.frame_data.popleft()
                predictions = self.predictor.get()
                yield predictions

            # When the last image is provided, we empty the buffer
            # to guarantee inference for all images and empty the result_queue
            if is_last_frame:
                while len(self.frame_data):
                    # Remove an entry to account for predicting one element form the task_queue
                    self.frame_data.popleft()
                    predictions = self.predictor.get()
                    yield predictions
        # Make a normal predictions
        else:
            yield self.predictor([current_frame])

    def start(self):
        """
        Used for starting the async prediction processes
        :return:
        """
        self.predictor.start()

    def shutdown(self):
        """
        Used putting the POISON_PILLS into subsequent processes and breaking the child processes
        :return:
        """
        self.predictor.shutdown()

    def join(self):
        """
        Used for terminating and joining the processes
        :return:
        """
        self.predictor.join()


class DemoDefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of given images.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `detectron_config.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `detectron_config.INPUT.FORMAT`.
    3. Apply resizing defined by `detectron_config.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a batch of input images and produce a batch of output.

    Examples:

    .. code-block:: python

        pred = DemoDefaultPredictor(detectron_config)
        inputs = cv2.imread("input.jpg")
        outputs = pred([inputs])
    """

    def __init__(self, cfg, detectron_config, img_height, img_width):
        """
        Initialize the predictor
        :param cfg: the demo cfg
        :param detectron_config: a prepared detectron2 config file
        :param img_height: (int) the image height of the original image
        :param img_width: (int) the image wight of the original image
        """
        self.cfg = cfg
        self.detectron_config = detectron_config.clone()  # detectron_config can be modified by model
        self.model = build_model(self.detectron_config)
        self.model.eval()
        self.metadata = MetadataCatalog.get(detectron_config.DATASETS.TEST[0])

        # Load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(detectron_config.MODEL.WEIGHTS)

        # Compute rescaling options once for speed improvement
        self.original_img_height = img_height
        self.original_img_width = img_width
        # The size of the short side and whether rescaling is required
        self.new_short_side, self.rescale_image = self.calculate_short_side()

        # used in filter_person_boxes
        self.cpu_device = torch.device("cpu")

        self.input_format = detectron_config.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def calculate_short_side(self):
        """
        Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
        If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
        Calculation is based on the original calculations
        :return:
            new_short_side (int): the short side to which our image will be rescaled
            rescale_image (bool): whether we have to do any rescaling at all
        """
        min_size = self.detectron_config.INPUT.MIN_SIZE_TEST
        max_size = self.detectron_config.INPUT.MAX_SIZE_TEST

        scale = min_size * 1.0 / min(self.original_img_height, self.original_img_width)
        if self.original_img_height < self.original_img_width:
            newh, neww = min_size, scale * self.original_img_width
        else:
            newh, neww = scale * self.original_img_height, min_size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        # The new short side to scale our images to
        new_short_side = min(neww, newh)

        # If the image is already correctly scaled, we do not have to rescale it
        rescale_image = min(self.original_img_height, self.original_img_width) != new_short_side

        return new_short_side, rescale_image

    def preprocess_images(self, imgs):
        """
        Preprocesses & resize imgs to fit the detection model input format.
        This is a faster method to preprocess images than in the original project:
        Preprocessing before took approx 0.015 seconds, now 0,006 --> improvement 50-80 %
        :param imgs: (list[np.ndarray]) a list of original images of shape (H, W, C) (in BGR order) and [0,255].
        :return:
            inputs: a list[{dicts}] of the input images with the following keys:
                image: tensor for a single image in (C, Hnew, Wnew), (BGR or RGB depending on model) and [0,255]
                height: the desired output height (original resolution)
                width: the desired output width (original resolution)
        """
        # Convert images to RGB if required
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

        # Resize image: short side to previously calculated short side
        if self.rescale_image:
            imgs = [cv2_transform.scale(self.new_short_side, img) for img in imgs]

        inputs = []
        for img in imgs:
            # Append inputs and (H, W, C) -> (C, H, W) and ndarray -> tensor
            inputs.append({"image": torch.as_tensor(img.astype("float32").transpose(2, 0, 1)),
                           "height": self.original_img_height, "width": self.original_img_width})

        return inputs

    def filter_person_boxes(self, model_output):
        """
        Time for one call with one output approx: 0.0002 s
        Filters the results for person detections
        https://detectron2.readthedocs.io/tutorials/models.html
        :param model_output: (list{dict}) See :doc:`/tutorials/models` for details about the format.
        :return:
            predictions (list[dict]): a list with an entry for each predicted image
                pred_boxes: ndarray of shape num_predictions, 4=the coordinates of the predicted boxes [x1, y1, x2, y2])
                scores: ndarray of shape (num_predictions) containing the confidence scores [0,1]
        """
        predictions = []
        for single_prediction in model_output:
            # Transfer relevant data to cpu
            single_prediction_cpu = single_prediction["instances"].to(self.cpu_device)._fields

            # List for storing whether the boxes are "high" enough
            height_selection_mask = []

            for box in single_prediction_cpu["pred_boxes"]:
                # Is the width (y2-y1) >= a predefined height
                height_selection_mask.append((box[3] - box[1]) >= self.cfg.DETECTRON.DEMO_MIN_BOX_HEIGHT)

            # List to tensor
            height_selection_mask = torch.BoolTensor(height_selection_mask)

            # Person has class label 0, we are only interested in this data
            person_selection_mask = single_prediction_cpu["pred_classes"] == 0

            # Use only boxes that are true for both criteria
            box_selection_mask = height_selection_mask & person_selection_mask


            # Append newly created dict with person boxes for current frame
            predictions.append(
                {"pred_boxes": single_prediction_cpu["pred_boxes"].tensor[box_selection_mask].data.numpy(),
                 "scores": single_prediction_cpu["scores"][box_selection_mask].data.numpy()})
        return predictions

    def __call__(self, original_images):
        """
        Computes the detections for a list of original images
        :param original_images: (list[np.ndarray]) a list of images of shape (H, W, C) (in BGR order) and [0,255]
        :return:
            predictions (list[dict]): a list with an entry for each predicted image
                pred_boxes: ndarray of shape num_predictions, 4 = the coordinates of the predicted boxes [x1, y1, x2, y2])
                scores: ndarray of shape (num_predictions) containing the confidence scores [0,1]
        """

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Preprocess the input images
            inputs = self.preprocess_images(original_images)

            # Compute the predicted detections
            #  See :doc:`/tutorials/models` for details about the format.
            model_output = self.model(inputs)

            return self.filter_person_boxes(model_output)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    Inference is based on the DemoDefaultPredictor, which requires passing the original image resolution
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, detectron_config, task_queue, result_queue, process_finished_queue,
                     img_height, img_width, batch_size):
            """
            A subprocess that uses the DemoDefaultPredictor
            to asynchronously predict images and store the results
            :param cfg: the dem cfg
            :param detectron_config: a prepared detectron2 config file
            :param task_queue: the task_queue of the async predictor
            :param result_queue: the result_queue of the async predictor
            :param process_finished_queue: if full, all _Predict_Worker processes finished successfully
            :param img_height: (int) the image height of the original image
            :param img_width: (int) the image width of the original image
            :param batch_size: (int) the batch size for the object predictor
            """
            self.cfg = cfg
            # Used as input for the DemoDefaultPredictor
            self.original_img_height = img_height
            self.original_img_width = img_width

            self.detectron_config = detectron_config
            self.task_queue = task_queue
            self.result_queue = result_queue
            self.process_finished_queue = process_finished_queue

            # Relevant for batch object detection
            self.batch_size = batch_size
            # Relevant for storing our variables
            self.batch_ids = []
            self.batch_image_data = []

            super().__init__()

        def run(self):
            """
            This method predicts the results for a list[images] (currently only 1) retreived
            from the task_queue and stores them in the result queue.
            Since we use multiprocessing, the order of the elements is not guaranteed and ordered in the get() function
            :return:
            """
            predictor = DemoDefaultPredictor(self.cfg, self.detectron_config, self.original_img_height, self.original_img_width)

            print(str(os.getpid()) + ": Object predictor started")

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    # Make prediction for remaining items
                    if len(self.batch_ids) > 0:
                        print(str(os.getpid()) + ": ##############Finishing batch of size: " + str(len(self.batch_ids)))
                        # Make final predictions
                        predictions = predictor(self.batch_image_data)
                        # and put the into the result queue
                        for img_idx, image, pred in zip(self.batch_ids, self.batch_image_data, predictions):
                            self.result_queue.put((img_idx, image, pred))

                    # Indicate that this process can be finished
                    print(str(os.getpid()) + ": ************Process can be finished")
                    self.process_finished_queue.put("FINISHED")

                    # Put poison pill and finish processes
                    while True:
                        # All processes have processed all data -> return and thus stop process
                        if self.process_finished_queue.full():
                            # Put poison pill to kill subsequent processes
                            self.result_queue.put((POISON_PILL, POISON_PILL, POISON_PILL))
                            print(str(os.getpid()) + ": Object predictor POISON Pill and break")
                            return
                        # Wait for other processes to finish
                        else:
                            time.sleep(0.1)

                # "Unzip" data and store it
                img_idx, image = task
                self.batch_ids.append(img_idx)
                self.batch_image_data.append(image)

                # Make predictions for full batch
                if len(self.batch_ids) == self.batch_size:
                    predictions = predictor(self.batch_image_data)

                    # print("-----------------")
                    # Update result_queue
                    for img_idx, image, pred in zip(self.batch_ids, self.batch_image_data, predictions):
                        # print(pred)
                        # print(type(pred))
                        self.result_queue.put((img_idx, image, pred))
                    # print("-----------------")

                    # Empty our data for next batch
                    self.batch_ids = []
                    self.batch_image_data = []

    def __init__(self, cfg, detectron_config, img_height, img_width,
                 num_gpus: int = 1, input_queue=None, output_queue=None, gpuid_action_recognition=None):
        """
        Initialize the AsyncPredictor
        :param cfg: the prototype config
        :param detectron_config: a prepared detectron2 config file
        :param img_height: (int) the image height of the original image
        :param img_width: (int) the image wight of the original image
        :param num_gpus: if 0, will run on CPU
        :param input_queue: (multiprocessing.queue) containing the input images
                    (img_idx, image of shape (H, W, C) (in BGR order) and [0,255])
        :param: output_queue: (multiprocessing.queue) containing the computed predictions #ToDo: explain format
        :param gpuid_action_recognition:  (int) the gpuid for object tracking (used for assigning batch sizes)
        """
        # Used as input for the DemoDefaultPredictor
        self.original_img_height = img_height
        self.original_img_width = img_width
        self.batch_size = cfg.DETECTRON.DEMO_BATCH_SIZE

        num_workers = max(num_gpus, 1)
        # A shared task queue that contains elements: (put_idx, image of shape (H, W, C) (in BGR order) and [0,255])
        self.task_queue = input_queue if input_queue else mp.Queue(maxsize=num_workers * 3)
        # A shared result queue that contains elements: (get_idx = put_idx of element, predictions (list{dict}))
        # The correct retreiving of the results is done using the get function
        self.result_queue = output_queue if output_queue else mp.Queue(maxsize=num_workers * 3)
        self.procs = []

        # Used to determine the right time for putting poison pills into the result_queue
        self.num_procs = max(num_gpus, 1)
        # If this queue is full, all child processes have successfully preprocessed all data
        self.process_finished_queue = mp.Queue(maxsize=self.num_procs)

        for gpuid in range(max(num_gpus, 1)):
            detectron_config = detectron_config.clone()
            detectron_config.defrost()
            detectron_config.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"

            # Determine the batch size -> we only use higher batch sizes for the gpus not occupied with
            # action prediction
            if gpuid == gpuid_action_recognition:
                batch_size_object_detection = 1
            # Also for our gpuid 0, we have to decrease the batch size, because we do deep_sort_tracking_here
            elif gpuid == 0:
                batch_size_object_detection = min(10, self.batch_size)
            else:
                batch_size_object_detection = self.batch_size

            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, detectron_config, self.task_queue, self.result_queue,
                                              self.process_finished_queue, self.original_img_height,
                                              self.original_img_width, batch_size_object_detection)
            )

        # used to distinguish images put in the task queue (unique and starting from 0)
        self.put_idx = -1
        # used to distinguish images retrieved from the  the result_queue (unique and starting from 0)
        self.get_idx = -1
        # A list that contains and is sorted by get_idxs (ascending) -> result_rank[0] is smallest get_idx
        self.result_rank = []
        # A list that contains the prediction results (predictions (list{dict})) and that is
        # also sorted by the get_idxs -> corresponding to result_rank
        self.result_data = []

    def start(self):
        """
        Start the processes
        :return:
        """
        for p in self.procs:
            p.start()
            print(str(os.getpid()) + ": Object_predictor started")

    def shutdown(self):
        """
        Put the stop signal into the queue, which leads to joining the other processes
        through posion pills
        :return:
        """
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())
            print(str(os.getpid()) + ": Object_predictor Poison pill triggered")

    def join(self):
        """
        Join the process(es)
        :return:
        """
        for p in self.procs:
            p.join()
            print(str(p.ident) + ": Object_predictor child process joined")

    def put(self, image):
        """
        Put an image with unique put_idx in the queue
        :param image: image of shape (H, W, C) (in BGR order) and [0,255]
        :return:
        """
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        """
        There are 2 possibilities when retrieving a prediction for an increasing get_idx:
        1. The prediction is already stored in result_date --> return result_data[0]
        2. The prediction has to be retrieved from the result_queue --> get item(s) from result queue.
                Since the result_queue is not ordered, we have to store
                results with idx > get_idx in result_rank and result_data at correct position
        Since the get_idx is increased by 1, we automatically get the images in the correct order!
        :return:
        """
        self.get_idx += 1  # the index needed for this request

        # 1. The prediction is already stored in result_date --> return result data
        # If the predictions for the current get_idx have already been extracted from the result_queue
        # stored in result_data, retrieve them and delete the data corresponding to
        # get_idx from the result_data and the result_rank
        # since get_idx is increasing, it is sufficient to check only the first element
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        # 2. The prediction has to be retrieved from the result_queue --> get item from result queue.
        while True:
            idx, res = self.result_queue.get()  # remove and return an item (in random order) from the result_queue
            # if the retreived idx corresponds to the get_idx, return the results
            if idx == self.get_idx:
                return res

            # if not we have the cae idx > get_idx --> we store the prediction for later use
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)
            # the index for inserting the element (starting from 0 as first position to insert)
            # bisect.bisect locates the insertion point in result_rank to keep a sorted order
            # defined by the get_idx. Thus, the element with the smallest get_idx is always at
            # the [0] position

    def __len__(self):
        """
        Returns how many already results have not been retrieved yet (probably not used)
        :return: int
        """
        return self.put_idx - self.get_idx

    def __call__(self, image):
        """
        I think, that this function is not used at all
        :param image:
        :return:
        """
        self.put(image)
        return self.get()

    @property
    def default_buffer_size(self):
        """
        Returns the default buffer size (for the result_queue) --> 5 for each process
        :return: the buffer size (int)
        """
        return len(self.procs) * 5
