# Created in the VideoProcessMining project based on https://github.com/jrosebr1/imutils/blob/master/imutils/video/filevideostream.py and https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py#L24

import slowfast.utils.logging as logging
from slowfast.utils.env import setup_environment
import os
import subprocess
from threading import Thread
import sys
import cv2
import time

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

logger = logging.get_logger(__name__)

"""
Our tools for extracting frames from video files
or webcam
"""

class FileVideoStream:
    """
    This class is based on imutils version for fast frame reading using queues
    https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    I fixed and error in the stop function and modified it for my multiprocessing use case
    """
    def __init__(self, path, transform=None, queue_size=128):
        """
        Initialize the class
        :param path: path to the video file
        :param transform: a function to transform the read images
        :param queue_size: the size of the queue
        """

        setup_environment()

        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # Get attributes of Video
        self.width = int(self.stream .get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream .get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames_per_second = self.stream .get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.stream .get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_length_seconds = self.get_video_length_in_seconds(path)

        # initialize the queue used to store frames read from
        # the video file
        self.video_image_queue = Queue(maxsize=queue_size)
        # The idx of an image starting from 0
        self.img_idx = -1
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        print(str(os.getpid()) + ": FileVideoStream started")
        return self

    def update(self):
        """
        Used to update the queue in a separate thread
        :return:
        """
        # keep looping infinitely
        print(str(os.getpid()) + ": Starting Update")
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                print(str(os.getpid()) + ": Break Update")
                break

            # otherwise, ensure the queue has room in it
            if not self.video_image_queue.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file and do not add an item to the queue
                if not grabbed:
                    self.stopped = True
                else:
                    # if there are transforms to be done, might as well
                    # do them on producer thread before handing back to
                    # consumer thread. ie. Usually the producer is so far
                    # ahead of consumer that we have time to spare.
                    #
                    # Python is not parallel but the transform operations
                    # are usually OpenCV native so release the GIL.
                    #
                    # Really just trying to avoid spinning up additional
                    # native threads and overheads of additional
                    # producer/consumer queues since this one was generally
                    # idle grabbing frames.
                    if self.transform:
                        frame = self.transform(frame)

                    # add the frame to the queue
                    self.img_idx += 1
                    self.video_image_queue.put((self.img_idx, frame))
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.video_image_queue.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.video_image_queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        # fixed error of none element
        return self.video_image_queue.qsize() > 0 and self.video_image_queue.queue[0] is not None

    def join(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
        print(str(os.getpid()) + ": FileVideoStream joined")

    def get_video_length_in_seconds(self, path):
        """
        :param path: path to the video file
        Returns the lenght of a video
        :return: length (float): seconds.milliseconds
        """
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)


class WebcamVideoStream:
    """
    # on the basis of imutils
    # https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py#L24
    """
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        print(str(os.getpid()) + ": FileVideoStream started")
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        print("FileVideoStream joined")
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True