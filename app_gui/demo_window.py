# Created in the VideoProcessMining project
import os
import traceback

import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot, QThreadPool
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import sys
import mimetypes

from slowfast.utils.misc import launch_job
from tools.test_net import test
from tools.train_net import train

import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
from tools.demo_net import run_demo
from tools.preprocess_net import create_folder_structure, extract_frames_from_videos_and_create_framelist_files, \
    compute_train_predict_box_list_and_create_file, compute_test_predict_boxes_and_create_file, compute_mean_and_std

FINISHED_PROGRESS_BAR_STYLE_SHEET = """
QProgressBar {
    border: 2px solid grey;
    border-radius: 5px;
}

QProgressBar::chunk {
    background-color: #92d050;
    width: 20px;
}
 """

class WorkerSignals(QObject):
    '''
    # see https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class DemoWindow(QMainWindow):
    """
    User Interface to access our prototype's capabilities
    To open the Qt Designer, activate your environment and type "designer" at the prompt and press enters
    """
    def __init__(self):
        super(DemoWindow, self).__init__()

        # Determines, whether buttons are enabled or disabled
        self.buttons_enabled = True

        # Used for multiprocessing
        self.threadpool = QThreadPool()
        # setup the UI components
        self.setupUi()

        self.path_to_current_cfg = get_path_to_current_cfg()
        if os.path.isfile(self.path_to_current_cfg):
            self.cfg = load_config(self.path_to_current_cfg)
            # Initialize the gui values with the config
            self.load_config_to_gui()

        # Register listeners
        self.register_config_listeners_and_changers()
        self.register_tool_listeners()


    def setupUi(self):
        self.setObjectName("self")
        self.resize(1130, 677)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.tab_tools = QtWidgets.QTabWidget(self.centralwidget)
        self.tab_tools.setGeometry(QtCore.QRect(21, 200, 1081, 351))
        self.tab_tools.setObjectName("tab_tools")
        self.preprocess_tab = QtWidgets.QWidget()
        self.preprocess_tab.setObjectName("preprocess_tab")
        self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files = QtWidgets.QToolButton(
            self.preprocess_tab)
        self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files.setGeometry(
            QtCore.QRect(10, 90, 341, 22))
        self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files.setObjectName(
            "preprocess_btn_extract_frames_from_videos_and_create_framelist_files")
        self.preprocess_label = QtWidgets.QLabel(self.preprocess_tab)
        self.preprocess_label.setGeometry(QtCore.QRect(10, 10, 711, 31))
        self.preprocess_label.setWordWrap(True)
        self.preprocess_label.setObjectName("preprocess_label")
        self.preprocess_btn_create_empty_annotation_files = QtWidgets.QToolButton(self.preprocess_tab)
        self.preprocess_btn_create_empty_annotation_files.setGeometry(QtCore.QRect(10, 50, 341, 22))
        self.preprocess_btn_create_empty_annotation_files.setObjectName("preprocess_btn_create_empty_annotation_files")
        self.preprocess_btn_compute_train_predict_box_list = QtWidgets.QToolButton(self.preprocess_tab)
        self.preprocess_btn_compute_train_predict_box_list.setGeometry(QtCore.QRect(10, 130, 341, 22))
        self.preprocess_btn_compute_train_predict_box_list.setObjectName(
            "preprocess_btn_compute_train_predict_box_list")
        self.preprocess_btn_compute_test_predict_boxes = QtWidgets.QToolButton(self.preprocess_tab)
        self.preprocess_btn_compute_test_predict_boxes.setGeometry(QtCore.QRect(10, 170, 341, 22))
        self.preprocess_btn_compute_test_predict_boxes.setObjectName("preprocess_btn_compute_test_predict_boxes")
        self.preprocess_label_2 = QtWidgets.QLabel(self.preprocess_tab)
        self.preprocess_label_2.setGeometry(QtCore.QRect(370, 130, 711, 31))
        self.preprocess_label_2.setWordWrap(True)
        self.preprocess_label_2.setObjectName("preprocess_label_2")
        self.preprocess_btn_compute_rgb_mean_and_std = QtWidgets.QToolButton(self.preprocess_tab)
        self.preprocess_btn_compute_rgb_mean_and_std.setGeometry(QtCore.QRect(10, 210, 341, 22))
        self.preprocess_btn_compute_rgb_mean_and_std.setObjectName("preprocess_btn_compute_rgb_mean_and_std")
        self.tab_tools.addTab(self.preprocess_tab, "")
        self.train_tab = QtWidgets.QWidget()
        self.train_tab.setObjectName("train_tab")
        self.train_groupBox_training_options = QtWidgets.QGroupBox(self.train_tab)
        self.train_groupBox_training_options.setGeometry(QtCore.QRect(10, 5, 1021, 51))
        self.train_groupBox_training_options.setObjectName("train_groupBox_training_options")
        self.train_label_EVAL_PERIOD = QtWidgets.QLabel(self.train_groupBox_training_options)
        self.train_label_EVAL_PERIOD.setGeometry(QtCore.QRect(12, 22, 301, 20))
        self.train_label_EVAL_PERIOD.setObjectName("train_label_EVAL_PERIOD")
        self.train_spinBox_EVAL_PERIOD = QtWidgets.QSpinBox(self.train_groupBox_training_options)
        self.train_spinBox_EVAL_PERIOD.setGeometry(QtCore.QRect(312, 20, 61, 24))
        self.train_spinBox_EVAL_PERIOD.setMinimum(1)
        self.train_spinBox_EVAL_PERIOD.setMaximum(1000)
        self.train_spinBox_EVAL_PERIOD.setObjectName("train_spinBox_EVAL_PERIOD")
        self.train_label_CHECKPOINT_PERIOD = QtWidgets.QLabel(self.train_groupBox_training_options)
        self.train_label_CHECKPOINT_PERIOD.setGeometry(QtCore.QRect(411, 22, 251, 20))
        self.train_label_CHECKPOINT_PERIOD.setObjectName("train_label_CHECKPOINT_PERIOD")
        self.train_spinBox_CHECKPOINT_PERIOD = QtWidgets.QSpinBox(self.train_groupBox_training_options)
        self.train_spinBox_CHECKPOINT_PERIOD.setGeometry(QtCore.QRect(657, 20, 61, 24))
        self.train_spinBox_CHECKPOINT_PERIOD.setMinimum(1)
        self.train_spinBox_CHECKPOINT_PERIOD.setMaximum(1000)
        self.train_spinBox_CHECKPOINT_PERIOD.setObjectName("train_spinBox_CHECKPOINT_PERIOD")
        self.train_spinBox_BATCH_SIZE = QtWidgets.QSpinBox(self.train_groupBox_training_options)
        self.train_spinBox_BATCH_SIZE.setGeometry(QtCore.QRect(880, 20, 61, 24))
        self.train_spinBox_BATCH_SIZE.setMinimum(1)
        self.train_spinBox_BATCH_SIZE.setMaximum(1000)
        self.train_spinBox_BATCH_SIZE.setObjectName("train_spinBox_BATCH_SIZE")
        self.train_label_BATCH_SIZE = QtWidgets.QLabel(self.train_groupBox_training_options)
        self.train_label_BATCH_SIZE.setGeometry(QtCore.QRect(758, 22, 121, 20))
        self.train_label_BATCH_SIZE.setObjectName("train_label_BATCH_SIZE")
        self.train_groupBox_transfer_learning = QtWidgets.QGroupBox(self.train_tab)
        self.train_groupBox_transfer_learning.setGeometry(QtCore.QRect(10, 69, 1021, 111))
        self.train_groupBox_transfer_learning.setObjectName("train_groupBox_transfer_learning")
        self.train_label_CHECKPOINT_FILE_PATH = QtWidgets.QLabel(self.train_groupBox_transfer_learning)
        self.train_label_CHECKPOINT_FILE_PATH.setGeometry(QtCore.QRect(8, 28, 321, 20))
        self.train_label_CHECKPOINT_FILE_PATH.setObjectName("train_label_CHECKPOINT_FILE_PATH")
        self.test_ceckbox_force_AUTO_RESUME = QtWidgets.QCheckBox(self.train_groupBox_transfer_learning)
        self.test_ceckbox_force_AUTO_RESUME.setGeometry(QtCore.QRect(10, 60, 911, 21))
        self.test_ceckbox_force_AUTO_RESUME.setObjectName("test_ceckbox_force_AUTO_RESUME")
        self.train_line_edit_CHECKPOINT_FILE_PATH = QtWidgets.QLineEdit(self.train_groupBox_transfer_learning)
        self.train_line_edit_CHECKPOINT_FILE_PATH.setGeometry(QtCore.QRect(330, 23, 661, 31))
        self.train_line_edit_CHECKPOINT_FILE_PATH.setText("")
        self.train_line_edit_CHECKPOINT_FILE_PATH.setObjectName("train_line_edit_CHECKPOINT_FILE_PATH")
        self.train_comboBox_CHECKPOINT_TYPE = QtWidgets.QComboBox(self.train_groupBox_transfer_learning)
        self.train_comboBox_CHECKPOINT_TYPE.setGeometry(QtCore.QRect(150, 90, 79, 16))
        self.train_comboBox_CHECKPOINT_TYPE.setObjectName("train_comboBox_CHECKPOINT_TYPE")
        self.train_comboBox_CHECKPOINT_TYPE.addItem("")
        self.train_comboBox_CHECKPOINT_TYPE.addItem("")
        self.train_label_CHECKPOINT_TYPE = QtWidgets.QLabel(self.train_groupBox_transfer_learning)
        self.train_label_CHECKPOINT_TYPE.setGeometry(QtCore.QRect(10, 88, 151, 16))
        self.train_label_CHECKPOINT_TYPE.setObjectName("train_label_CHECKPOINT_TYPE")
        self.train_groupBox_finetuning = QtWidgets.QGroupBox(self.train_tab)
        self.train_groupBox_finetuning.setGeometry(QtCore.QRect(10, 190, 1021, 41))
        self.train_groupBox_finetuning.setObjectName("train_groupBox_finetuning")
        self.train_ceckbox_FINETUNE = QtWidgets.QCheckBox(self.train_groupBox_finetuning)
        self.train_ceckbox_FINETUNE.setGeometry(QtCore.QRect(10, 21, 201, 21))
        self.train_ceckbox_FINETUNE.setObjectName("train_ceckbox_FINETUNE")
        self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE = QtWidgets.QCheckBox(self.train_groupBox_finetuning)
        self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE.setGeometry(QtCore.QRect(250, 20, 680, 21))
        self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE.setObjectName(
            "train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE")
        self.train_groupBox_solver_and_model = QtWidgets.QGroupBox(self.train_tab)
        self.train_groupBox_solver_and_model.setGeometry(QtCore.QRect(10, 240, 821, 71))
        self.train_groupBox_solver_and_model.setObjectName("train_groupBox_solver_and_model")
        self.train_doubleSpinBoxd_BASE_LR = QtWidgets.QDoubleSpinBox(self.train_groupBox_solver_and_model)
        self.train_doubleSpinBoxd_BASE_LR.setGeometry(QtCore.QRect(435, 49, 81, 16))
        self.train_doubleSpinBoxd_BASE_LR.setDecimals(6)
        self.train_doubleSpinBoxd_BASE_LR.setSingleStep(1e-06)
        self.train_doubleSpinBoxd_BASE_LR.setObjectName("train_doubleSpinBoxd_BASE_LR")
        self.train_label_BASE_LR = QtWidgets.QLabel(self.train_groupBox_solver_and_model)
        self.train_label_BASE_LR.setGeometry(QtCore.QRect(312, 49, 121, 16))
        self.train_label_BASE_LR.setObjectName("train_label_BASE_LR")
        self.train_comboBox_HEAD_ACT = QtWidgets.QComboBox(self.train_groupBox_solver_and_model)
        self.train_comboBox_HEAD_ACT.setGeometry(QtCore.QRect(211, 28, 79, 16))
        self.train_comboBox_HEAD_ACT.setObjectName("train_comboBox_HEAD_ACT")
        self.train_comboBox_HEAD_ACT.addItem("")
        self.train_comboBox_HEAD_ACT.addItem("")
        self.train_label_HEAD_ACT = QtWidgets.QLabel(self.train_groupBox_solver_and_model)
        self.train_label_HEAD_ACT.setGeometry(QtCore.QRect(10, 28, 201, 16))
        self.train_label_HEAD_ACT.setObjectName("train_label_HEAD_ACT")
        self.train_label_LOSS_FUNC = QtWidgets.QLabel(self.train_groupBox_solver_and_model)
        self.train_label_LOSS_FUNC.setGeometry(QtCore.QRect(10, 50, 201, 16))
        self.train_label_LOSS_FUNC.setObjectName("train_label_LOSS_FUNC")
        self.train_comboBox_LOSS_FUNC = QtWidgets.QComboBox(self.train_groupBox_solver_and_model)
        self.train_comboBox_LOSS_FUNC.setGeometry(QtCore.QRect(211, 50, 79, 16))
        self.train_comboBox_LOSS_FUNC.setObjectName("train_comboBox_LOSS_FUNC")
        self.train_comboBox_LOSS_FUNC.addItem("")
        self.train_comboBox_LOSS_FUNC.addItem("")
        self.train_spinBox_MAX_EPOCH = QtWidgets.QSpinBox(self.train_groupBox_solver_and_model)
        self.train_spinBox_MAX_EPOCH.setGeometry(QtCore.QRect(539, 24, 61, 24))
        self.train_spinBox_MAX_EPOCH.setMinimum(1)
        self.train_spinBox_MAX_EPOCH.setMaximum(1000)
        self.train_spinBox_MAX_EPOCH.setObjectName("train_spinBox_MAX_EPOCH")
        self.train_label_MAX_EPOCH = QtWidgets.QLabel(self.train_groupBox_solver_and_model)
        self.train_label_MAX_EPOCH.setGeometry(QtCore.QRect(310, 26, 221, 20))
        self.train_label_MAX_EPOCH.setObjectName("train_label_MAX_EPOCH")
        self.train_label_WARMUP_EPOCHS = QtWidgets.QLabel(self.train_groupBox_solver_and_model)
        self.train_label_WARMUP_EPOCHS.setGeometry(QtCore.QRect(611, 26, 101, 20))
        self.train_label_WARMUP_EPOCHS.setObjectName("train_label_WARMUP_EPOCHS")
        self.train_doubleSpinBox_WARMUP_EPOCHS = QtWidgets.QDoubleSpinBox(self.train_groupBox_solver_and_model)
        self.train_doubleSpinBox_WARMUP_EPOCHS.setGeometry(QtCore.QRect(720, 24, 61, 24))
        self.train_doubleSpinBox_WARMUP_EPOCHS.setDecimals(1)
        self.train_doubleSpinBox_WARMUP_EPOCHS.setMaximum(10000.0)
        self.train_doubleSpinBox_WARMUP_EPOCHS.setObjectName("train_doubleSpinBox_WARMUP_EPOCHS")
        self.train_btn_start_train = QtWidgets.QPushButton(self.train_tab)
        self.train_btn_start_train.setGeometry(QtCore.QRect(865, 288, 181, 23))
        self.train_btn_start_train.setObjectName("train_btn_start_train")
        self.tab_tools.addTab(self.train_tab, "")
        self.test_tab = QtWidgets.QWidget()
        self.test_tab.setObjectName("test_tab")
        self.test_groupBox_test_options = QtWidgets.QGroupBox(self.test_tab)
        self.test_groupBox_test_options.setGeometry(QtCore.QRect(10, 10, 1021, 91))
        self.test_groupBox_test_options.setObjectName("test_groupBox_test_options")
        self.test_spinBox_BATCH_SIZE = QtWidgets.QSpinBox(self.test_groupBox_test_options)
        self.test_spinBox_BATCH_SIZE.setGeometry(QtCore.QRect(140, 20, 61, 24))
        self.test_spinBox_BATCH_SIZE.setMinimum(1)
        self.test_spinBox_BATCH_SIZE.setMaximum(1000)
        self.test_spinBox_BATCH_SIZE.setObjectName("test_spinBox_BATCH_SIZE")
        self.test_spinBox_BATCH_SIZE.setDisabled(True)
        self.test_label_BATCH_SIZE = QtWidgets.QLabel(self.test_groupBox_test_options)
        self.test_label_BATCH_SIZE.setGeometry(QtCore.QRect(18, 22, 121, 20))
        self.test_label_BATCH_SIZE.setObjectName("test_label_BATCH_SIZE")
        self.test_line_edit_CHECKPOINT_FILE_PATH = QtWidgets.QLineEdit(self.test_groupBox_test_options)
        self.test_line_edit_CHECKPOINT_FILE_PATH.setGeometry(QtCore.QRect(332, 55, 661, 31))
        self.test_line_edit_CHECKPOINT_FILE_PATH.setText("")
        self.test_line_edit_CHECKPOINT_FILE_PATH.setObjectName("test_line_edit_CHECKPOINT_FILE_PATH")
        self.test_label_CHECKPOINT_FILE_PATH = QtWidgets.QLabel(self.test_groupBox_test_options)
        self.test_label_CHECKPOINT_FILE_PATH.setGeometry(QtCore.QRect(10, 60, 321, 20))
        self.test_label_CHECKPOINT_FILE_PATH.setObjectName("test_label_CHECKPOINT_FILE_PATH")
        self.test_comboBox_CHECKPOINT_TYPE = QtWidgets.QComboBox(self.test_groupBox_test_options)
        self.test_comboBox_CHECKPOINT_TYPE.setGeometry(QtCore.QRect(470, 23, 79, 16))
        self.test_comboBox_CHECKPOINT_TYPE.setObjectName("test_comboBox_CHECKPOINT_TYPE")
        self.test_comboBox_CHECKPOINT_TYPE.addItem("")
        self.test_comboBox_CHECKPOINT_TYPE.addItem("")
        self.test_label_CHECKPOINT_TYPE = QtWidgets.QLabel(self.test_groupBox_test_options)
        self.test_label_CHECKPOINT_TYPE.setGeometry(QtCore.QRect(330, 21, 151, 16))
        self.test_label_CHECKPOINT_TYPE.setObjectName("test_label_CHECKPOINT_TYPE")
        self.test_btn_start_test = QtWidgets.QPushButton(self.test_tab)
        self.test_btn_start_test.setGeometry(QtCore.QRect(858, 286, 181, 23))
        self.test_btn_start_test.setObjectName("test_btn_start_test")
        self.test_label_info = QtWidgets.QLabel(self.test_tab)
        self.test_label_info.setGeometry(QtCore.QRect(20, 110, 891, 16))
        self.test_label_info.setObjectName("test_label_info")
        self.tab_tools.addTab(self.test_tab, "")
        self.demo_tab = QtWidgets.QWidget()
        self.demo_tab.setObjectName("demo_tab")
        self.demo_groupBox_video_options = QtWidgets.QGroupBox(self.demo_tab)
        self.demo_groupBox_video_options.setGeometry(QtCore.QRect(20, 6, 1011, 80))
        self.demo_groupBox_video_options.setObjectName("demo_groupBox_video_options")
        self.demo_checkBox_show_video = QtWidgets.QCheckBox(self.demo_groupBox_video_options)
        self.demo_checkBox_show_video.setGeometry(QtCore.QRect(10, 23, 181, 21))
        self.demo_checkBox_show_video.setObjectName("demo_checkBox_show_video")
        self.demo_checkBox_show_video_debugging_info = QtWidgets.QCheckBox(self.demo_groupBox_video_options)
        self.demo_checkBox_show_video_debugging_info.setGeometry(QtCore.QRect(751, 21, 221, 21))
        self.demo_checkBox_show_video_debugging_info.setObjectName("demo_checkBox_show_video_debugging_info")
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH = QtWidgets.QDoubleSpinBox(
            self.demo_groupBox_video_options)
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setGeometry(QtCore.QRect(600, 50, 61, 24))
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setMaximum(1.0)
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setSingleStep(0.01)
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setObjectName(
            "demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH")
        self.demo_label_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH = QtWidgets.QLabel(self.demo_groupBox_video_options)
        self.demo_label_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setGeometry(QtCore.QRect(280, 50, 321, 20))
        self.demo_label_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setObjectName(
            "demo_label_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH")
        self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE = QtWidgets.QCheckBox(self.demo_groupBox_video_options)
        self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE.setGeometry(QtCore.QRect(10, 50, 251, 21))
        self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE.setObjectName("demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE")
        self.demo_label_display_VIDEO_DISPLAY_SCALING_FACTOR = QtWidgets.QLabel(self.demo_groupBox_video_options)
        self.demo_label_display_VIDEO_DISPLAY_SCALING_FACTOR.setGeometry(QtCore.QRect(280, 21, 261, 20))
        self.demo_label_display_VIDEO_DISPLAY_SCALING_FACTOR.setObjectName(
            "demo_label_display_VIDEO_DISPLAY_SCALING_FACTOR")
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR = QtWidgets.QDoubleSpinBox(
            self.demo_groupBox_video_options)
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.setGeometry(QtCore.QRect(600, 20, 61, 24))
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.setMaximum(1.0)
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.setSingleStep(0.01)
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.setObjectName(
            "demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR")
        self.demo_btn_start_demo = QtWidgets.QPushButton(self.demo_tab)
        self.demo_btn_start_demo.setGeometry(QtCore.QRect(850, 290, 181, 23))
        self.demo_btn_start_demo.setObjectName("demo_btn_start_demo")
        self.demo_progressBar = QtWidgets.QProgressBar(self.demo_tab)
        self.demo_progressBar.setGeometry(QtCore.QRect(20, 248, 1011, 23))
        self.demo_progressBar.setProperty("value", 0)
        self.demo_progressBar.setObjectName("demo_progressBar")
        self.demo_groupBox_detectron2_options = QtWidgets.QGroupBox(self.demo_tab)
        self.demo_groupBox_detectron2_options.setGeometry(QtCore.QRect(20, 174, 1011, 61))
        self.demo_groupBox_detectron2_options.setObjectName("demo_groupBox_detectron2_options")
        self.demo_label_detectron2_person_score_thresh = QtWidgets.QLabel(self.demo_groupBox_detectron2_options)
        self.demo_label_detectron2_person_score_thresh.setGeometry(QtCore.QRect(12, 28, 251, 20))
        self.demo_label_detectron2_person_score_thresh.setObjectName("demo_label_detectron2_person_score_thresh")
        self.demo_doubleSpinBox_detectron2_person_score_thresh = QtWidgets.QDoubleSpinBox(
            self.demo_groupBox_detectron2_options)
        self.demo_doubleSpinBox_detectron2_person_score_thresh.setGeometry(QtCore.QRect(263, 26, 61, 24))
        self.demo_doubleSpinBox_detectron2_person_score_thresh.setMaximum(1.0)
        self.demo_doubleSpinBox_detectron2_person_score_thresh.setSingleStep(0.01)
        self.demo_doubleSpinBox_detectron2_person_score_thresh.setObjectName(
            "demo_doubleSpinBox_detectron2_person_score_thresh")
        self.demo_label_detectron2_batch_size = QtWidgets.QLabel(self.demo_groupBox_detectron2_options)
        self.demo_label_detectron2_batch_size.setGeometry(QtCore.QRect(714, 30, 191, 20))
        self.demo_label_detectron2_batch_size.setObjectName("demo_label_detectron2_batch_size")
        self.demo_spinBox_detectron2_batch_size = QtWidgets.QSpinBox(self.demo_groupBox_detectron2_options)
        self.demo_spinBox_detectron2_batch_size.setGeometry(QtCore.QRect(920, 30, 61, 24))
        self.demo_spinBox_detectron2_batch_size.setMaximum(12)
        self.demo_spinBox_detectron2_batch_size.setObjectName("demo_spinBox_detectron2_batch_size")
        self.demo_label_DEMO_MIN_BOX_HEIGHT = QtWidgets.QLabel(self.demo_groupBox_detectron2_options)
        self.demo_label_DEMO_MIN_BOX_HEIGHT.setGeometry(QtCore.QRect(340, 28, 281, 20))
        self.demo_label_DEMO_MIN_BOX_HEIGHT.setObjectName("demo_label_DEMO_MIN_BOX_HEIGHT")
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT = QtWidgets.QSpinBox(self.demo_groupBox_detectron2_options)
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.setGeometry(QtCore.QRect(630, 26, 61, 24))
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.setMaximum(2000)
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.setSingleStep(1)
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.setObjectName("demo_spinBox_DEMO_MIN_BOX_HEIGHT")
        self.demo_label_video_file = QtWidgets.QLabel(self.demo_tab)
        self.demo_label_video_file.setGeometry(QtCore.QRect(210, 290, 541, 16))
        self.demo_label_video_file.setObjectName("demo_label_video_file")
        self.demo_btn_select_video_file = QtWidgets.QToolButton(self.demo_tab)
        self.demo_btn_select_video_file.setGeometry(QtCore.QRect(22, 286, 161, 22))
        self.demo_btn_select_video_file.setObjectName("demo_btn_select_video_file")
        self.demo_groupBox_export_options = QtWidgets.QGroupBox(self.demo_tab)
        self.demo_groupBox_export_options.setGeometry(QtCore.QRect(20, 97, 1011, 61))
        self.demo_groupBox_export_options.setObjectName("demo_groupBox_export_options")
        self.demo_label_export_score = QtWidgets.QLabel(self.demo_groupBox_export_options)
        self.demo_label_export_score.setGeometry(QtCore.QRect(588, 30, 341, 20))
        self.demo_label_export_score.setObjectName("demo_label_export_score")
        self.demo_doubleSpinBox_export_action_recognition_score_tresh = QtWidgets.QDoubleSpinBox(self.demo_groupBox_export_options)
        self.demo_doubleSpinBox_export_action_recognition_score_tresh.setGeometry(QtCore.QRect(930, 30, 61, 24))
        self.demo_doubleSpinBox_export_action_recognition_score_tresh.setMaximum(1.0)
        self.demo_doubleSpinBox_export_action_recognition_score_tresh.setSingleStep(0.01)
        self.demo_doubleSpinBox_export_action_recognition_score_tresh.setObjectName("demo_doubleSpinBox_export_action_recognition_score_tresh")
        self.demo_checkBox_EXPORT_EXPORT_RESULTS = QtWidgets.QCheckBox(self.demo_groupBox_export_options)
        self.demo_checkBox_EXPORT_EXPORT_RESULTS.setGeometry(QtCore.QRect(20, 30, 191, 21))
        self.demo_checkBox_EXPORT_EXPORT_RESULTS.setObjectName("demo_checkBox_EXPORT_EXPORT_RESULTS")
        self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE = QtWidgets.QCheckBox(self.demo_groupBox_export_options)
        self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE.setGeometry(QtCore.QRect(231, 30, 341, 21))
        self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE.setObjectName("demo_checkBox_MULTIPLE_ACTION_POSSIBLE")
        self.tab_tools.addTab(self.demo_tab, "")
        self.plainTextEdit_edit_log_console = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_edit_log_console.setGeometry(QtCore.QRect(21, 570, 1081, 51))
        self.plainTextEdit_edit_log_console.setPlainText("")
        self.plainTextEdit_edit_log_console.setObjectName("plainTextEdit_edit_log_console")
        self.tab_deep_learning = QtWidgets.QTabWidget(self.centralwidget)
        self.tab_deep_learning.setGeometry(QtCore.QRect(24, 10, 1081, 171))
        self.tab_deep_learning.setObjectName("tab_deep_learning")
        self.current_config_tab = QtWidgets.QWidget()
        self.current_config_tab.setObjectName("current_config_tab")
        self.current_config_btn_select_other_cfg = QtWidgets.QToolButton(self.current_config_tab)
        self.current_config_btn_select_other_cfg.setGeometry(QtCore.QRect(890, 20, 181, 22))
        self.current_config_btn_select_other_cfg.setObjectName("current_config_btn_select_other_cfg")
        self.current_config_btn_save_changes_to_current_cfg = QtWidgets.QToolButton(self.current_config_tab)
        self.current_config_btn_save_changes_to_current_cfg.setGeometry(QtCore.QRect(656, 110, 201, 22))
        self.current_config_btn_save_changes_to_current_cfg.setObjectName(
            "current_config_btn_save_changes_to_current_cfg")
        self.current_config_label_path_to_current_config = QtWidgets.QLabel(self.current_config_tab)
        self.current_config_label_path_to_current_config.setGeometry(QtCore.QRect(10, 16, 181, 16))
        self.current_config_label_path_to_current_config.setObjectName("current_config_label_path_to_current_config")
        self.current_config_label_path_to_current_dataset_folder = QtWidgets.QLabel(self.current_config_tab)
        self.current_config_label_path_to_current_dataset_folder.setGeometry(QtCore.QRect(10, 42, 181, 16))
        self.current_config_label_path_to_current_dataset_folder.setObjectName(
            "current_config_label_path_to_current_dataset_folder")
        self.current_config_btn_save_changes_to_new_cfg = QtWidgets.QToolButton(self.current_config_tab)
        self.current_config_btn_save_changes_to_new_cfg.setGeometry(QtCore.QRect(868, 110, 201, 22))
        self.current_config_btn_save_changes_to_new_cfg.setObjectName("current_config_btn_save_changes_to_new_cfg")
        self.current_config_label_path_to_current_config_value = QtWidgets.QLabel(self.current_config_tab)
        self.current_config_label_path_to_current_config_value.setGeometry(QtCore.QRect(211, 6, 671, 31))
        self.current_config_label_path_to_current_config_value.setWordWrap(True)
        self.current_config_label_path_to_current_config_value.setObjectName(
            "current_config_label_path_to_current_config_value")
        self.current_config_label_path_to_current_dataset_folder_value = QtWidgets.QLabel(self.current_config_tab)
        self.current_config_label_path_to_current_dataset_folder_value.setGeometry(QtCore.QRect(210, 33, 681, 31))
        self.current_config_label_path_to_current_dataset_folder_value.setWordWrap(True)
        self.current_config_label_path_to_current_dataset_folder_value.setObjectName(
            "current_config_label_path_to_current_dataset_folder_value")
        self.current_config_btn_select_other_dataset_folder = QtWidgets.QToolButton(self.current_config_tab)
        self.current_config_btn_select_other_dataset_folder.setGeometry(QtCore.QRect(890, 50, 181, 22))
        self.current_config_btn_select_other_dataset_folder.setObjectName(
            "current_config_btn_select_other_dataset_folder")
        self.current_config_groupBox_hardware_options = QtWidgets.QGroupBox(self.current_config_tab)
        self.current_config_groupBox_hardware_options.setGeometry(QtCore.QRect(10, 70, 611, 61))
        self.current_config_groupBox_hardware_options.setObjectName("current_config_groupBox_hardware_options")
        self.current_config_spinBox_NUM_GPUS = QtWidgets.QSpinBox(self.current_config_groupBox_hardware_options)
        self.current_config_spinBox_NUM_GPUS.setGeometry(QtCore.QRect(145, 28, 61, 24))
        self.current_config_spinBox_NUM_GPUS.setMinimum(1)
        self.current_config_spinBox_NUM_GPUS.setMaximum(100)
        self.current_config_spinBox_NUM_GPUS.setObjectName("current_config_spinBox_NUM_GPUS")
        self.current_config_label_NUM_GPUS = QtWidgets.QLabel(self.current_config_groupBox_hardware_options)
        self.current_config_label_NUM_GPUS.setGeometry(QtCore.QRect(10, 30, 131, 20))
        self.current_config_label_NUM_GPUS.setObjectName("current_config_label_NUM_GPUS")
        self.current_config_label_NUM_WORKERS = QtWidgets.QLabel(self.current_config_groupBox_hardware_options)
        self.current_config_label_NUM_WORKERS.setGeometry(QtCore.QRect(235, 32, 201, 20))
        self.current_config_label_NUM_WORKERS.setObjectName("current_config_label_NUM_WORKERS")
        self.current_config_spinBox_NUM_WORKERS = QtWidgets.QSpinBox(self.current_config_groupBox_hardware_options)
        self.current_config_spinBox_NUM_WORKERS.setGeometry(QtCore.QRect(450, 30, 61, 24))
        self.current_config_spinBox_NUM_WORKERS.setMaximum(100)
        self.current_config_spinBox_NUM_WORKERS.setObjectName("current_config_spinBox_NUM_WORKERS")
        self.tab_deep_learning.addTab(self.current_config_tab, "")
        self.Custom_Dataset = QtWidgets.QWidget()
        self.Custom_Dataset.setObjectName("Custom_Dataset")
        self.custom_dataset_checkBox_bgr = QtWidgets.QCheckBox(self.Custom_Dataset)
        self.custom_dataset_checkBox_bgr.setGeometry(QtCore.QRect(20, 110, 211, 21))
        self.custom_dataset_checkBox_bgr.setObjectName("custom_dataset_checkBox_bgr")
        self.custom_dataset_comboBox_image_processing_backend = QtWidgets.QComboBox(self.Custom_Dataset)
        self.custom_dataset_comboBox_image_processing_backend.setGeometry(QtCore.QRect(390, 110, 79, 16))
        self.custom_dataset_comboBox_image_processing_backend.setObjectName(
            "custom_dataset_comboBox_image_processing_backend")
        self.custom_dataset_comboBox_image_processing_backend.addItem("")
        self.custom_dataset_comboBox_image_processing_backend.addItem("")
        self.custom_dataset_label_image_processing_backend = QtWidgets.QLabel(self.Custom_Dataset)
        self.custom_dataset_label_image_processing_backend.setGeometry(QtCore.QRect(220, 110, 191, 20))
        self.custom_dataset_label_image_processing_backend.setObjectName(
            "custom_dataset_label_image_processing_backend")
        self.custom_dataset_label_frame_rate = QtWidgets.QLabel(self.Custom_Dataset)
        self.custom_dataset_label_frame_rate.setGeometry(QtCore.QRect(20, 10, 201, 16))
        self.custom_dataset_label_frame_rate.setObjectName("custom_dataset_label_frame_rate")
        self.custom_dataset_spinBoxd_frame_rate = QtWidgets.QSpinBox(self.Custom_Dataset)
        self.custom_dataset_spinBoxd_frame_rate.setGeometry(QtCore.QRect(240, 10, 66, 16))
        self.custom_dataset_spinBoxd_frame_rate.setMinimum(1)
        self.custom_dataset_spinBoxd_frame_rate.setMaximum(120)
        self.custom_dataset_spinBoxd_frame_rate.setSingleStep(1)
        self.custom_dataset_spinBoxd_frame_rate.setObjectName("custom_dataset_spinBoxd_frame_rate")
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh = QtWidgets.QDoubleSpinBox(
            self.Custom_Dataset)
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.setGeometry(
            QtCore.QRect(670, 50, 61, 24))
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.setMaximum(1.0)
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.setSingleStep(0.01)
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.setObjectName(
            "custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh")
        self.custom_dataset_label_custom_dataset_detection_score_thresh = QtWidgets.QLabel(self.Custom_Dataset)
        self.custom_dataset_label_custom_dataset_detection_score_thresh.setGeometry(QtCore.QRect(20, 40, 651, 41))
        self.custom_dataset_label_custom_dataset_detection_score_thresh.setWordWrap(True)
        self.custom_dataset_label_custom_dataset_detection_score_thresh.setObjectName(
            "custom_dataset_label_custom_dataset_detection_score_thresh")
        self.tab_deep_learning.addTab(self.Custom_Dataset, "")
        self.action_recognizer_tab = QtWidgets.QWidget()
        self.action_recognizer_tab.setObjectName("action_recognizer_tab")
        self.action_recognizer_label_ckeckpoint_file_path = QtWidgets.QLabel(self.action_recognizer_tab)
        self.action_recognizer_label_ckeckpoint_file_path.setGeometry(QtCore.QRect(30, 28, 271, 16))
        self.action_recognizer_label_ckeckpoint_file_path.setObjectName("action_recognizer_label_ckeckpoint_file_path")
        self.action_recognizer_label_checkpoint_type = QtWidgets.QLabel(self.action_recognizer_tab)
        self.action_recognizer_label_checkpoint_type.setGeometry(QtCore.QRect(330, 81, 151, 16))
        self.action_recognizer_label_checkpoint_type.setObjectName("action_recognizer_label_checkpoint_type")
        self.action_recognizer_line_edit_checkpoint_file_path = QtWidgets.QLineEdit(self.action_recognizer_tab)
        self.action_recognizer_line_edit_checkpoint_file_path.setGeometry(QtCore.QRect(330, 30, 691, 31))
        self.action_recognizer_line_edit_checkpoint_file_path.setText("")
        self.action_recognizer_line_edit_checkpoint_file_path.setObjectName(
            "action_recognizer_line_edit_checkpoint_file_path")
        self.action_recognizer_comboBox_checkpoint_type = QtWidgets.QComboBox(self.action_recognizer_tab)
        self.action_recognizer_comboBox_checkpoint_type.setGeometry(QtCore.QRect(470, 83, 79, 16))
        self.action_recognizer_comboBox_checkpoint_type.setObjectName("action_recognizer_comboBox_checkpoint_type")
        self.action_recognizer_comboBox_checkpoint_type.addItem("")
        self.action_recognizer_comboBox_checkpoint_type.addItem("")
        self.action_recognizer_spinBox_NUM_CLASSES = QtWidgets.QSpinBox(self.action_recognizer_tab)
        self.action_recognizer_spinBox_NUM_CLASSES.setGeometry(QtCore.QRect(206, 78, 61, 24))
        self.action_recognizer_spinBox_NUM_CLASSES.setMinimum(1)
        self.action_recognizer_spinBox_NUM_CLASSES.setMaximum(1000)
        self.action_recognizer_spinBox_NUM_CLASSES.setObjectName("action_recognizer_spinBox_NUM_CLASSES")
        self.action_recognizer_label_NUM_CLASSES = QtWidgets.QLabel(self.action_recognizer_tab)
        self.action_recognizer_label_NUM_CLASSES.setGeometry(QtCore.QRect(32, 80, 181, 20))
        self.action_recognizer_label_NUM_CLASSES.setObjectName("action_recognizer_label_NUM_CLASSES")
        self.tab_deep_learning.addTab(self.action_recognizer_tab, "")
        self.detectron2_tab = QtWidgets.QWidget()
        self.detectron2_tab.setObjectName("detectron2_tab")
        self.detectron2_label_detection_model_cfg = QtWidgets.QLabel(self.detectron2_tab)
        self.detectron2_label_detection_model_cfg.setGeometry(QtCore.QRect(10, 10, 171, 16))
        self.detectron2_label_detection_model_cfg.setObjectName("detectron2_label_detection_model_cfg")
        self.detectron2_label_info = QtWidgets.QLabel(self.detectron2_tab)
        self.detectron2_label_info.setGeometry(QtCore.QRect(20, 120, 751, 20))
        self.detectron2_label_info.setOpenExternalLinks(True)
        self.detectron2_label_info.setObjectName("detectron2_label_info")
        self.detectron2_label_model_weights = QtWidgets.QLabel(self.detectron2_tab)
        self.detectron2_label_model_weights.setGeometry(QtCore.QRect(10, 59, 171, 16))
        self.detectron2_label_model_weights.setObjectName("detectron2_label_model_weights")
        self.detectron2_line_edit_detection_model_cfg = QtWidgets.QLineEdit(self.detectron2_tab)
        self.detectron2_line_edit_detection_model_cfg.setGeometry(QtCore.QRect(170, 10, 861, 41))
        self.detectron2_line_edit_detection_model_cfg.setText("")
        self.detectron2_line_edit_detection_model_cfg.setObjectName("detectron2_line_edit_detection_model_cfg")
        self.detectron2_line_edit_model_weights = QtWidgets.QLineEdit(self.detectron2_tab)
        self.detectron2_line_edit_model_weights.setGeometry(QtCore.QRect(170, 60, 861, 41))
        self.detectron2_line_edit_model_weights.setText("")
        self.detectron2_line_edit_model_weights.setObjectName("detectron2_line_edit_model_weights")
        self.tab_deep_learning.addTab(self.detectron2_tab, "")
        self.deepsort_tab = QtWidgets.QWidget()
        self.deepsort_tab.setObjectName("deepsort_tab")
        self.deep_sort_label_reid_ckpt = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_reid_ckpt.setGeometry(QtCore.QRect(20, 10, 121, 16))
        self.deep_sort_label_reid_ckpt.setObjectName("deep_sort_label_reid_ckpt")
        self.deep_sort_label_max_dist = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_max_dist.setGeometry(QtCore.QRect(20, 30, 121, 16))
        self.deep_sort_label_max_dist.setObjectName("deep_sort_label_max_dist")
        self.deep_sort_label_min_confidence = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_min_confidence.setGeometry(QtCore.QRect(20, 50, 121, 16))
        self.deep_sort_label_min_confidence.setObjectName("deep_sort_label_min_confidence")
        self.deep_sort_label_nms_max_overlap = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_nms_max_overlap.setGeometry(QtCore.QRect(20, 70, 121, 16))
        self.deep_sort_label_nms_max_overlap.setObjectName("deep_sort_label_nms_max_overlap")
        self.deep_sort_label_max_iou_distance = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_max_iou_distance.setGeometry(QtCore.QRect(20, 90, 121, 16))
        self.deep_sort_label_max_iou_distance.setObjectName("deep_sort_label_max_iou_distance")
        self.deep_sort_label_max_age = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_max_age.setGeometry(QtCore.QRect(20, 110, 121, 16))
        self.deep_sort_label_max_age.setObjectName("deep_sort_label_max_age")
        self.deep_sort_label_n_init = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_n_init.setGeometry(QtCore.QRect(270, 110, 121, 16))
        self.deep_sort_label_n_init.setObjectName("deep_sort_label_n_init")
        self.deep_sort_label_nn_budget = QtWidgets.QLabel(self.deepsort_tab)
        self.deep_sort_label_nn_budget.setGeometry(QtCore.QRect(470, 110, 121, 16))
        self.deep_sort_label_nn_budget.setObjectName("deep_sort_label_nn_budget")
        self.deep_sort_line_edit_reid_ckpt = QtWidgets.QLineEdit(self.deepsort_tab)
        self.deep_sort_line_edit_reid_ckpt.setGeometry(QtCore.QRect(160, 10, 851, 16))
        self.deep_sort_line_edit_reid_ckpt.setText("")
        self.deep_sort_line_edit_reid_ckpt.setObjectName("deep_sort_line_edit_reid_ckpt")
        self.deep_sort_doubleSpinBoxd_max_dist = QtWidgets.QDoubleSpinBox(self.deepsort_tab)
        self.deep_sort_doubleSpinBoxd_max_dist.setGeometry(QtCore.QRect(160, 30, 66, 16))
        self.deep_sort_doubleSpinBoxd_max_dist.setSingleStep(0.02)
        self.deep_sort_doubleSpinBoxd_max_dist.setObjectName("deep_sort_doubleSpinBoxd_max_dist")
        self.deep_sort_doubleSpinBoxd_min_confidence = QtWidgets.QDoubleSpinBox(self.deepsort_tab)
        self.deep_sort_doubleSpinBoxd_min_confidence.setGeometry(QtCore.QRect(160, 50, 66, 16))
        self.deep_sort_doubleSpinBoxd_min_confidence.setMaximum(1.0)
        self.deep_sort_doubleSpinBoxd_min_confidence.setSingleStep(0.01)
        self.deep_sort_doubleSpinBoxd_min_confidence.setObjectName("deep_sort_doubleSpinBoxd_min_confidence")
        self.deep_sort_doubleSpinBoxd_nms_max_overlap = QtWidgets.QDoubleSpinBox(self.deepsort_tab)
        self.deep_sort_doubleSpinBoxd_nms_max_overlap.setGeometry(QtCore.QRect(160, 70, 66, 16))
        self.deep_sort_doubleSpinBoxd_nms_max_overlap.setMaximum(1.0)
        self.deep_sort_doubleSpinBoxd_nms_max_overlap.setSingleStep(0.01)
        self.deep_sort_doubleSpinBoxd_nms_max_overlap.setObjectName("deep_sort_doubleSpinBoxd_nms_max_overlap")
        self.deep_sort_doubleSpinBoxd_max_iou_distance = QtWidgets.QDoubleSpinBox(self.deepsort_tab)
        self.deep_sort_doubleSpinBoxd_max_iou_distance.setGeometry(QtCore.QRect(160, 90, 66, 16))
        self.deep_sort_doubleSpinBoxd_max_iou_distance.setMaximum(15.0)
        self.deep_sort_doubleSpinBoxd_max_iou_distance.setSingleStep(0.1)
        self.deep_sort_doubleSpinBoxd_max_iou_distance.setObjectName("deep_sort_doubleSpinBoxd_max_iou_distance")
        self.deep_sort_spinBoxd_max_age = QtWidgets.QSpinBox(self.deepsort_tab)
        self.deep_sort_spinBoxd_max_age.setGeometry(QtCore.QRect(160, 110, 66, 16))
        self.deep_sort_spinBoxd_max_age.setMinimum(1)
        self.deep_sort_spinBoxd_max_age.setMaximum(10000)
        self.deep_sort_spinBoxd_max_age.setSingleStep(2)
        self.deep_sort_spinBoxd_max_age.setObjectName("deep_sort_spinBoxd_max_age")
        self.deep_sort_spinBoxd_n_init = QtWidgets.QSpinBox(self.deepsort_tab)
        self.deep_sort_spinBoxd_n_init.setGeometry(QtCore.QRect(330, 110, 66, 16))
        self.deep_sort_spinBoxd_n_init.setMinimum(1)
        self.deep_sort_spinBoxd_n_init.setMaximum(10000)
        self.deep_sort_spinBoxd_n_init.setSingleStep(2)
        self.deep_sort_spinBoxd_n_init.setObjectName("deep_sort_spinBoxd_n_init")
        self.deep_sort_spinBoxd_nn_budget = QtWidgets.QSpinBox(self.deepsort_tab)
        self.deep_sort_spinBoxd_nn_budget.setGeometry(QtCore.QRect(550, 110, 66, 16))
        self.deep_sort_spinBoxd_nn_budget.setMinimum(1)
        self.deep_sort_spinBoxd_nn_budget.setMaximum(10000)
        self.deep_sort_spinBoxd_nn_budget.setSingleStep(5)
        self.deep_sort_spinBoxd_nn_budget.setObjectName("deep_sort_spinBoxd_nn_budget")
        self.tab_deep_learning.addTab(self.deepsort_tab, "")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1130, 20))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        self.tab_tools.setCurrentIndex(0)
        self.tab_deep_learning.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "VideoProcessMining"))
        self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files.setText(
            _translate("MainWindow", "Extract frames from videos and create framelist files"))
        self.preprocess_label.setText(_translate("MainWindow",
                                                 "Please follow the dataset preparation instructions in order to acheive successfull preprocessing"))
        self.preprocess_btn_create_empty_annotation_files.setText(
            _translate("MainWindow", "Create folder structure and empty annotation files"))
        self.preprocess_btn_compute_train_predict_box_list.setText(
            _translate("MainWindow", "Predict additional bounding boxes for training"))
        self.preprocess_btn_compute_test_predict_boxes.setText(
            _translate("MainWindow", "Predict bounding boxes for test"))
        self.preprocess_label_2.setText(_translate("MainWindow",
                                                   "Make sure to manually filter out boxes with low IoU values and remove header from file"))
        self.preprocess_btn_compute_rgb_mean_and_std.setText(_translate("MainWindow", "Compute RGB mean and std"))
        self.tab_tools.setTabText(self.tab_tools.indexOf(self.preprocess_tab), _translate("MainWindow", "Preprocess"))
        self.train_groupBox_training_options.setTitle(_translate("MainWindow", "Training Options"))
        self.train_label_EVAL_PERIOD.setText(
            _translate("MainWindow", "Evaluate model on test data every period epochs"))
        self.train_label_CHECKPOINT_PERIOD.setText(_translate("MainWindow", "Create Checkpoint every period epochs"))
        self.train_label_BATCH_SIZE.setText(_translate("MainWindow", "Batch Size training"))
        self.train_groupBox_transfer_learning.setTitle(
            _translate("MainWindow", "Transfer learning (model is trained from scratch if no options is chosen)"))
        self.train_label_CHECKPOINT_FILE_PATH.setText(
            _translate("MainWindow", "Use the following model as base model for training"))
        self.test_ceckbox_force_AUTO_RESUME.setText(_translate("MainWindow",
                                                               "If training is interrupted, resume training using model with highest epoch in \"checkpoints\" folder (in this case set checkpoint type to \"pytorch\")"))
        self.train_comboBox_CHECKPOINT_TYPE.setItemText(0, _translate("MainWindow", "caffe2"))
        self.train_comboBox_CHECKPOINT_TYPE.setItemText(1, _translate("MainWindow", "pytorch"))
        self.train_label_CHECKPOINT_TYPE.setText(_translate("MainWindow", "Checkpoint Type"))
        self.train_groupBox_finetuning.setTitle(_translate("MainWindow",
                                                           "Finetuning (head only, please adjust config, if you want to unfreeze other layers as well)"))
        self.train_ceckbox_FINETUNE.setText(_translate("MainWindow", "Use finetuning (head only)"))
        self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE.setText(_translate("MainWindow",
                                                                                "Force Training Epoch back to 1 (useful when training complete model after finetuning head only is finished)"))
        self.train_groupBox_solver_and_model.setTitle(_translate("MainWindow", "Model and solver options"))
        self.train_label_BASE_LR.setText(_translate("MainWindow", "Base learning rate"))
        self.train_comboBox_HEAD_ACT.setItemText(0, _translate("MainWindow", "softmax"))
        self.train_comboBox_HEAD_ACT.setItemText(1, _translate("MainWindow", "sigmoid"))
        self.train_label_HEAD_ACT.setText(_translate("MainWindow", "Model head activation function"))
        self.train_label_LOSS_FUNC.setText(_translate("MainWindow", "Loss function"))
        self.train_comboBox_LOSS_FUNC.setItemText(0, _translate("MainWindow", "bce"))
        self.train_comboBox_LOSS_FUNC.setItemText(1, _translate("MainWindow", "cross_entropy"))
        self.train_label_MAX_EPOCH.setText(_translate("MainWindow", "Maximum number of training epochs"))
        self.train_label_WARMUP_EPOCHS.setText(_translate("MainWindow", "Warump epochs"))
        self.train_btn_start_train.setText(_translate("MainWindow", "Start Training"))
        self.tab_tools.setTabText(self.tab_tools.indexOf(self.train_tab), _translate("MainWindow", "Train"))
        self.test_groupBox_test_options.setTitle(_translate("MainWindow", "Test Options"))
        self.test_label_BATCH_SIZE.setText(_translate("MainWindow", "Batch Size test"))
        self.test_label_CHECKPOINT_FILE_PATH.setText(
            _translate("MainWindow", "Load model weights from the following path"))
        self.test_comboBox_CHECKPOINT_TYPE.setItemText(0, _translate("MainWindow", "caffe2"))
        self.test_comboBox_CHECKPOINT_TYPE.setItemText(1, _translate("MainWindow", "pytorch"))
        self.test_label_CHECKPOINT_TYPE.setText(_translate("MainWindow", "Checkpoint Type"))
        self.test_btn_start_test.setText(_translate("MainWindow", "Start Test"))
        self.test_label_info.setText(_translate("MainWindow",
                                                "If you define no path, the prototype always uses the model with the highest epoch that can be found in the \"checkpoints\" folder"))
        self.tab_tools.setTabText(self.tab_tools.indexOf(self.test_tab), _translate("MainWindow", "Test"))
        self.demo_groupBox_video_options.setTitle(_translate("MainWindow", "Modify the video related export options"))
        self.demo_checkBox_show_video.setText(_translate("MainWindow", "Annotate and show video"))
        self.demo_checkBox_show_video_debugging_info.setText(_translate("MainWindow", "Display technical information"))
        self.demo_label_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setText(
            _translate("MainWindow", "Annotate only activity classes with min certainty of"))
        self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE.setText(
            _translate("MainWindow", "Export annotated video as video file"))
        self.demo_label_display_VIDEO_DISPLAY_SCALING_FACTOR.setText(
            _translate("MainWindow", "Scale display resolution of video by factor"))
        self.demo_btn_start_demo.setText(_translate("MainWindow", "Start Demo"))
        self.demo_groupBox_detectron2_options.setTitle(
            _translate("MainWindow", "Modify your person detection options (affects Detectron2)"))
        self.demo_label_detectron2_person_score_thresh.setText(
            _translate("MainWindow", "Only use bounding boxes with score >="))
        self.demo_label_detectron2_batch_size.setText(_translate("MainWindow", "Batch size for person detection"))
        self.demo_label_DEMO_MIN_BOX_HEIGHT.setText(
            _translate("MainWindow", "Only use bounding boxes with pixel height >="))
        self.demo_label_video_file.setText(_translate("MainWindow", "No video file selected...."))
        self.demo_btn_select_video_file.setText(_translate("MainWindow", "Select video file"))
        self.demo_groupBox_export_options.setTitle(_translate("MainWindow", "Modify your XES export options"))
        self.demo_label_export_score.setText(
            _translate("MainWindow", "Set min score for an activity to be recognized as event"))
        self.demo_checkBox_EXPORT_EXPORT_RESULTS.setText(_translate("MainWindow", "Export XES log and csv files"))
        self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE.setText(
            _translate("MainWindow", "An actor can perform several activities concurrently"))
        self.tab_tools.setTabText(self.tab_tools.indexOf(self.demo_tab), _translate("MainWindow", "Demo"))
        self.tab_deep_learning.setToolTip(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.current_config_btn_select_other_cfg.setText(_translate("MainWindow", "Select other config"))
        self.current_config_btn_save_changes_to_current_cfg.setText(
            _translate("MainWindow", "Save changes to current config"))
        self.current_config_label_path_to_current_config.setText(_translate("MainWindow", "Path to current config"))
        self.current_config_label_path_to_current_dataset_folder.setText(
            _translate("MainWindow", "Path to current dataset folder"))
        self.current_config_btn_save_changes_to_new_cfg.setText(_translate("MainWindow", "Save changes to new config"))
        self.current_config_label_path_to_current_config_value.setText(
            _translate("MainWindow", "path to current config"))
        self.current_config_label_path_to_current_dataset_folder_value.setText(
            _translate("MainWindow", "path to current dataset folder"))
        self.current_config_btn_select_other_dataset_folder.setText(
            _translate("MainWindow", "Select other dataset folder"))
        self.current_config_groupBox_hardware_options.setTitle(
            _translate("MainWindow", "Hardware options for preprocess, train, test, demo"))
        self.current_config_label_NUM_GPUS.setText(_translate("MainWindow", "Max number of GPUs"))
        self.current_config_label_NUM_WORKERS.setText(_translate("MainWindow", "Number of workers dataloader"))
        self.tab_deep_learning.setTabText(self.tab_deep_learning.indexOf(self.current_config_tab),
                                          _translate("MainWindow", "Current Configuration"))
        self.custom_dataset_checkBox_bgr.setText(_translate("MainWindow", "Image Data is in BGR order"))
        self.custom_dataset_comboBox_image_processing_backend.setItemText(0, _translate("MainWindow", "cv2"))
        self.custom_dataset_comboBox_image_processing_backend.setItemText(1, _translate("MainWindow", "pytorch"))
        self.custom_dataset_label_image_processing_backend.setText(_translate("MainWindow", "Image processing backend"))
        self.custom_dataset_label_frame_rate.setText(_translate("MainWindow", "Targeted Frame Rate"))
        self.custom_dataset_label_custom_dataset_detection_score_thresh.setText(_translate("MainWindow",
                                                                                           "To be considered for preprocessing, training, and test bounding boxes have to have a min score of "))
        self.tab_deep_learning.setTabText(self.tab_deep_learning.indexOf(self.Custom_Dataset),
                                          _translate("MainWindow", "Custom Dataset"))
        self.action_recognizer_label_ckeckpoint_file_path.setText(
            _translate("MainWindow", "Checkpoint file path to pretrained model"))
        self.action_recognizer_label_checkpoint_type.setText(_translate("MainWindow", "Checkpoint Type"))
        self.action_recognizer_comboBox_checkpoint_type.setItemText(0, _translate("MainWindow", "pytorch"))
        self.action_recognizer_comboBox_checkpoint_type.setItemText(1, _translate("MainWindow", "caffe2"))
        self.action_recognizer_label_NUM_CLASSES.setText(_translate("MainWindow", "Number of activity classes"))
        self.tab_deep_learning.setTabText(self.tab_deep_learning.indexOf(self.action_recognizer_tab),
                                          _translate("MainWindow", "Activity Recognizer"))
        self.detectron2_label_detection_model_cfg.setText(_translate("MainWindow", "Detection Model Config"))
        self.detectron2_label_info.setText(_translate("MainWindow",
                                                      "<html><head/><body><p>It is possible to adjust these value<span style=\" color:#000000;\">s based on a given model id (current model id</span><span style=\" font-family:\'DejaVu Sans Mono\'; color:#000000;\"> 137849458</span><span style=\" font-family:\'DejaVu Sans Mono\'; color:#808080;\">)</span> from <a href=\"https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md\"><span style=\" text-decoration: underline; color:#0000ff;\">Detectron2Github</span></a></p></body></html>"))
        self.detectron2_label_model_weights.setText(_translate("MainWindow", "Model Weights"))
        self.tab_deep_learning.setTabText(self.tab_deep_learning.indexOf(self.detectron2_tab),
                                          _translate("MainWindow", "Detectron2"))
        self.deep_sort_label_reid_ckpt.setText(_translate("MainWindow", "REID-CHECKPOINT"))
        self.deep_sort_label_max_dist.setText(_translate("MainWindow", "MAX_DIST"))
        self.deep_sort_label_min_confidence.setText(_translate("MainWindow", "MIN CONFIDENCE"))
        self.deep_sort_label_nms_max_overlap.setText(_translate("MainWindow", "NMS MAX OVERLAP"))
        self.deep_sort_label_max_iou_distance.setText(_translate("MainWindow", "MAX IOU_DISTANCE"))
        self.deep_sort_label_max_age.setText(_translate("MainWindow", "MAX AGE"))
        self.deep_sort_label_n_init.setText(_translate("MainWindow", "N INIT"))
        self.deep_sort_label_nn_budget.setText(_translate("MainWindow", "NN Budget"))
        self.tab_deep_learning.setTabText(self.tab_deep_learning.indexOf(self.deepsort_tab),
                                          _translate("MainWindow", "Deep Sort"))


    def enable_or_disable_all_buttons(self):
        """
        Disables all buttons
        :return:
        """
        if self.buttons_enabled:
            self.buttons_enabled = False
        else:
            self.buttons_enabled = True

        # Enable or disable the buttons
        self.current_config_btn_select_other_cfg.setEnabled(self.buttons_enabled)
        self.preprocess_btn_compute_rgb_mean_and_std.setEnabled(self.buttons_enabled)
        self.preprocess_btn_compute_test_predict_boxes.setEnabled(self.buttons_enabled)
        self.preprocess_btn_compute_train_predict_box_list.setEnabled(self.buttons_enabled)
        self.preprocess_btn_create_empty_annotation_files.setEnabled(self.buttons_enabled)
        self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files.setEnabled(self.buttons_enabled)
        self.demo_btn_select_video_file.setEnabled(self.buttons_enabled)
        self.demo_btn_start_demo.setEnabled(self.buttons_enabled)
        self.train_btn_start_train.setEnabled(self.buttons_enabled)
        self.test_btn_start_test.setEnabled(self.buttons_enabled)

    def register_config_listeners_and_changers(self):
        """
        Registers the listener functions that adjust our self.cfg values as soon as user input changes
        :return:
        """

        # For current configuration
        self.current_config_btn_select_other_cfg.clicked.connect(self.current_config_btn_select_other_cfg_clicked)
        self.current_config_btn_select_other_dataset_folder.clicked.connect(self.current_config_btn_select_other_dataset_folder_clicked)
        self.current_config_btn_save_changes_to_current_cfg.clicked.connect(self.current_config_btn_save_changes_to_current_cfg_clicked)
        self.current_config_btn_save_changes_to_new_cfg.clicked.connect(self.current_config_btn_save_changes_to_new_cfg_clicked)
        self.current_config_spinBox_NUM_GPUS.valueChanged.connect(
            self.current_config_spinBox_NUM_GPUS_value_changed)
        self.current_config_spinBox_NUM_WORKERS.valueChanged.connect(
            self.current_config_spinBox_NUM_WORKERS_value_changed)

        # For custom dataset
        self.custom_dataset_checkBox_bgr.stateChanged.connect(self.custom_dataset_checkBox_bgr_state_changed)
        self.custom_dataset_comboBox_image_processing_backend.currentIndexChanged.connect(self.custom_dataset_comboBox_image_processing_backend_current_index_changed)
        self.custom_dataset_spinBoxd_frame_rate.valueChanged.connect(
            self.custom_dataset_spinBoxd_frame_rate_value_changed)
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.valueChanged.connect(self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh_value_changed)

        # For Activity Recognizer
        self.action_recognizer_line_edit_checkpoint_file_path.textChanged.connect(self.action_recognizer_line_edit_checkpoint_file_path_text_changed)
        self.action_recognizer_comboBox_checkpoint_type.currentIndexChanged.connect(self.action_recognizer_comboBox_checkpoint_type_current_index_changed)
        self.action_recognizer_spinBox_NUM_CLASSES.valueChanged.connect(self.action_recognizer_spinBox_NUM_CLASSES_value_changed)

        # For Detectron2
        self.detectron2_line_edit_detection_model_cfg.textChanged.connect(self.detectron2_line_edit_detection_model_cfg_text_changed)
        self.detectron2_line_edit_model_weights.textChanged.connect(self.detectron2_line_edit_model_weights_text_changed)

        # For Deep Sort
        self.deep_sort_line_edit_reid_ckpt.textChanged.connect(self.deep_sort_line_edit_reid_ckpt_text_changed)
        self.deep_sort_doubleSpinBoxd_max_dist.valueChanged.connect(self.deep_sort_doubleSpinBoxd_max_dist_value_changed)
        self.deep_sort_doubleSpinBoxd_min_confidence.valueChanged.connect(self.deep_sort_doubleSpinBoxd_min_confidence_value_changed)
        self.deep_sort_doubleSpinBoxd_nms_max_overlap.valueChanged.connect(self.deep_sort_doubleSpinBoxd_nms_max_overlap_value_changed)
        self.deep_sort_doubleSpinBoxd_max_iou_distance.valueChanged.connect(self.deep_sort_doubleSpinBoxd_max_iou_distance_value_changed)
        self.deep_sort_spinBoxd_max_age.valueChanged.connect(self.deep_sort_spinBoxd_max_age_value_changed)
        self.deep_sort_spinBoxd_n_init.valueChanged.connect(self.deep_sort_spinBoxd_n_init_value_changed)
        self.deep_sort_spinBoxd_nn_budget.valueChanged.connect(self.deep_sort_spinBoxd_nn_budget_value_changed)

        # For train
        self.train_spinBox_EVAL_PERIOD.valueChanged.connect(self.train_spinBox_EVAL_PERIOD_value_changed)
        self.train_spinBox_CHECKPOINT_PERIOD.valueChanged.connect(self.train_spinBox_CHECKPOINT_PERIOD_value_changed)
        self.train_spinBox_BATCH_SIZE.valueChanged.connect(self.train_spinBox_BATCH_SIZE_value_changed)
        self.train_line_edit_CHECKPOINT_FILE_PATH.textChanged.connect(self.train_line_edit_CHECKPOINT_FILE_PATH_text_changed)
        self.test_ceckbox_force_AUTO_RESUME.stateChanged.connect(self.test_ceckbox_force_AUTO_RESUME_state_changed)
        self.train_comboBox_CHECKPOINT_TYPE.currentIndexChanged.connect(self.train_comboBox_CHECKPOINT_TYPE_current_index_changed)
        self.train_ceckbox_FINETUNE.stateChanged.connect(self.train_ceckbox_FINETUNE_state_changed)
        self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE.stateChanged.connect(self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE_state_changed)
        self.train_comboBox_HEAD_ACT.currentIndexChanged.connect(self.train_comboBox_HEAD_ACT_current_index_changed)
        self.train_spinBox_MAX_EPOCH.valueChanged.connect(self.train_spinBox_MAX_EPOCH_value_changed)
        self.train_doubleSpinBox_WARMUP_EPOCHS.valueChanged.connect(self.train_doubleSpinBox_WARMUP_EPOCHS_value_changed)
        self.train_comboBox_LOSS_FUNC.currentIndexChanged.connect(self.train_comboBox_LOSS_FUNC_current_index_changed)
        self.train_doubleSpinBoxd_BASE_LR.valueChanged.connect(self.train_doubleSpinBoxd_BASE_LR_value_changed)

        #For test
        self.test_line_edit_CHECKPOINT_FILE_PATH.textChanged.connect(self.test_line_edit_CHECKPOINT_FILE_PATH_text_changed)
        self.test_comboBox_CHECKPOINT_TYPE.currentIndexChanged.connect(self.test_comboBox_CHECKPOINT_TYPE_current_index_changed)

        # For demo
        self.demo_checkBox_show_video.stateChanged.connect(self.demo_checkBox_show_video_state_changed)
        self.demo_checkBox_show_video_debugging_info.stateChanged.connect(self.demo_checkBox_show_video_debugging_info_state_changed)
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.valueChanged.connect(self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH_value_changed)
        self.demo_doubleSpinBox_export_action_recognition_score_tresh.valueChanged.connect(self.demo_doubleSpinBox_export_action_recognition_score_tresh_value_changed)
        self.demo_spinBox_detectron2_batch_size.valueChanged.connect(self.demo_spinBox_detectron2_batch_size_value_changed)
        self.demo_doubleSpinBox_detectron2_person_score_thresh.valueChanged.connect(
            self.demo_doubleSpinBox_detectron2_person_score_thresh_value_changed)
        self.demo_btn_select_video_file.clicked.connect(self.demo_btn_select_video_file_clicked)
        self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE.stateChanged.connect(self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE_state_changed)
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.valueChanged.connect(
            self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR_value_changed)
        self.demo_checkBox_EXPORT_EXPORT_RESULTS.stateChanged.connect(
            self.demo_checkBox_EXPORT_EXPORT_RESULTS_state_changed)
        self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE.stateChanged.connect(
            self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE_state_changed)
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.valueChanged.connect(
            self.demo_spinBox_DEMO_MIN_BOX_HEIGHT_value_changed)

    def register_tool_listeners(self):
        """
        Registers the functions that call our tool functions (i.e. preprocess, train, test, demo)
        :return:
        """

        # For demo
        self.demo_btn_start_demo.clicked.connect(self.demo_btn_start_demo_clicked)

        # For Preprocess
        self.preprocess_btn_create_empty_annotation_files.clicked.connect(self.preprocess_btn_create_empty_annotation_files_clicked)
        self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files.clicked.connect(
            self.preprocess_btn_extract_frames_from_videos_and_create_framelist_files_clicked)
        self.preprocess_btn_compute_train_predict_box_list.clicked.connect(
            self.preprocess_btn_compute_train_predict_box_list_clicked)
        self.preprocess_btn_compute_test_predict_boxes.clicked.connect(
            self.preprocess_btn_compute_test_predict_boxes_clicked)
        self.preprocess_btn_compute_rgb_mean_and_std.clicked.connect(
            self.preprocess_btn_compute_rgb_mean_and_std_clicked)

        # For train
        self.train_btn_start_train.clicked.connect(self.train_btn_start_train_clicked)

        # For test
        self.test_btn_start_test.clicked.connect(self.test_btn_start_test_clicked)

    def load_config_to_gui(self):
        """
        Load the initial values from the config file
        :return:
        """
        # For Current configuration:
        self.current_config_label_path_to_current_config_value.setText(self.path_to_current_cfg)
        self.current_config_label_path_to_current_dataset_folder_value.setText(self.cfg.OUTPUT_DIR)

        self.cfg.NUM_GPUS = min(self.cfg.NUM_GPUS, torch.cuda.device_count())
        self.current_config_spinBox_NUM_GPUS.setValue(self.cfg.NUM_GPUS)
        self.current_config_spinBox_NUM_GPUS.setMaximum(torch.cuda.device_count())
        self.current_config_spinBox_NUM_WORKERS.setValue(self.cfg.DATA_LOADER.NUM_WORKERS)

        # For Custom Dataset
        self.custom_dataset_checkBox_bgr.setChecked(self.cfg.CUSTOM_DATASET.BGR)
        self.custom_dataset_comboBox_image_processing_backend.setCurrentText(self.cfg.CUSTOM_DATASET.IMG_PROC_BACKEND)
        self.custom_dataset_spinBoxd_frame_rate.setValue(self.cfg.CUSTOM_DATASET.FRAME_RATE)
        self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.setValue(self.cfg.CUSTOM_DATASET.DETECTION_SCORE_THRESH)

        # For Activity Recognizer
        self.action_recognizer_line_edit_checkpoint_file_path.setText(self.cfg.ACTIONRECOGNIZER.CHECKPOINT_FILE_PATH)
        self.action_recognizer_comboBox_checkpoint_type.setCurrentText(self.cfg.ACTIONRECOGNIZER.CHECKPOINT_TYPE)
        self.action_recognizer_spinBox_NUM_CLASSES.setValue(self.cfg.MODEL.NUM_CLASSES)

        # For Detectron2
        self.detectron2_line_edit_detection_model_cfg.setText(self.cfg.DETECTRON.DETECTION_MODEL_CFG)
        self.detectron2_line_edit_model_weights.setText(self.cfg.DETECTRON.MODEL_WEIGHTS)

        # For Demo
        self.demo_checkBox_show_video.setChecked(self.cfg.DEMO.VIDEO_SHOW_VIDEO_ENABLE)
        self.demo_checkBox_show_video_debugging_info.setChecked(self.cfg.DEMO.VIDEO_SHOW_VIDEO_DEBUGGING_INFO)
        self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.setValue(self.cfg.DEMO.VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH)
        self.demo_doubleSpinBox_export_action_recognition_score_tresh.setValue(self.cfg.DEMO.EXPORT_MIN_CATEGORY_EXPORT_SCORE)
        self.demo_spinBox_detectron2_batch_size.setValue(self.cfg.DETECTRON.DEMO_BATCH_SIZE)
        self.demo_doubleSpinBox_detectron2_person_score_thresh.setValue(self.cfg.DETECTRON.DEMO_PERSON_SCORE_THRESH)
        self.demo_label_video_file.setText(self.cfg.DEMO.VIDEO_SOURCE_PATH)

        self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE.setChecked(self.cfg.CUSTOM_DATASET.MULTIPLE_ACTION_POSSIBLE)
        self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE.setChecked(self.cfg.DEMO.VIDEO_EXPORT_VIDEO_ENABLE)
        self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.setValue(self.cfg.DEMO.VIDEO_DISPLAY_SCALING_FACTOR)
        self.demo_checkBox_EXPORT_EXPORT_RESULTS.setChecked(self.cfg.DEMO.EXPORT_EXPORT_RESULTS)
        self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.setValue(self.cfg.DETECTRON.DEMO_MIN_BOX_HEIGHT)

        # For Deep Sort
        self.deep_sort_line_edit_reid_ckpt.setText(self.cfg.DEMO.DEEPSORT_REID_CKPT)
        self.deep_sort_doubleSpinBoxd_max_dist.setValue(self.cfg.DEMO.DEEPSORT_MAX_DIST)
        self.deep_sort_doubleSpinBoxd_min_confidence.setValue(self.cfg.DEMO.DEEPSORT_MIN_CONFIDENCE)
        self.deep_sort_doubleSpinBoxd_nms_max_overlap.setValue(self.cfg.DEMO.DEEPSORT_NMS_MAX_OVERLAP)
        self.deep_sort_doubleSpinBoxd_max_iou_distance.setValue(self.cfg.DEMO.DEEPSORT_MAX_IOU_DISTANCE)
        self.deep_sort_spinBoxd_max_age.setValue(self.cfg.DEMO.DEEPSORT_MAX_AGE)
        self.deep_sort_spinBoxd_n_init.setValue(self.cfg.DEMO.DEEPSORT_N_INIT)
        self.deep_sort_spinBoxd_nn_budget.setValue(self.cfg.DEMO.DEEPSORT_NN_BUDGET)

        #For Train
        self.train_spinBox_EVAL_PERIOD.setValue(self.cfg.TRAIN.EVAL_PERIOD)
        self.train_spinBox_CHECKPOINT_PERIOD.setValue(self.cfg.TRAIN.CHECKPOINT_PERIOD)
        self.update_train_batch_size(self.cfg.TRAIN.BATCH_SIZE)
        self.train_line_edit_CHECKPOINT_FILE_PATH.setText(self.cfg.TRAIN.CHECKPOINT_FILE_PATH)
        self.test_ceckbox_force_AUTO_RESUME.setChecked(self.cfg.TRAIN.AUTO_RESUME)
        self.train_comboBox_CHECKPOINT_TYPE.setCurrentText(self.cfg.TRAIN.CHECKPOINT_TYPE)
        self.train_ceckbox_FINETUNE.setChecked(self.cfg.TRAIN.FINETUNE)
        self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE.setChecked(self.cfg.TRAIN.FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE)
        self.train_comboBox_HEAD_ACT.setCurrentText(self.cfg.MODEL.HEAD_ACT)
        self.train_spinBox_MAX_EPOCH.setValue(self.cfg.SOLVER.MAX_EPOCH)
        self.train_doubleSpinBox_WARMUP_EPOCHS.setValue(self.cfg.SOLVER.WARMUP_EPOCHS)
        self.train_comboBox_LOSS_FUNC.setCurrentText(self.cfg.MODEL.LOSS_FUNC)
        self.train_doubleSpinBoxd_BASE_LR.setValue(self.cfg.SOLVER.BASE_LR)

        #For Test
        self.update_test_batch_size()
        self.test_line_edit_CHECKPOINT_FILE_PATH.setText(self.cfg.TEST.CHECKPOINT_FILE_PATH)
        self.test_comboBox_CHECKPOINT_TYPE.setCurrentText(self.cfg.TEST.CHECKPOINT_TYPE)

        # Tensorboard
        self.plainTextEdit_edit_log_console.setPlainText("To visualize your training and test process use the following command:\ntensorboard  --logdir " + os.path.join(self.cfg.OUTPUT_DIR, "runs-custom"))

    ######### Functionalities for running the demo

    def demo_btn_start_demo_clicked(self):
        """
        Start the demo in our worker thread
        with the current config values, and adjust progress bar accordingly
        """
        confirmed = showDialog("Start Demo",
                               "Please start the demo only with a valid checkpoint file path in the Activity Recognizer tab")
        if confirmed:
            # Reset the progress bar
            self.reset_demo_progress_bar()

            # Pass the function to execute
            worker = Worker(self.start_demo_from_worker)  # Any other args, kwargs are passed to the run function
            worker.signals.finished.connect(self.set_demo_progress_bar_finished)
            worker.signals.progress.connect(self.update_demo_progress_bar)

            # Execute
            self.threadpool.start(worker)

    def start_demo_from_worker(self, progress_callback):
        """
        is the function used by the worker, see demo_btn_start_demo_clicked
        :param progress_callback: (pyqtSignal) used for signaling back the progress from the demo
        """
        self.enable_or_disable_all_buttons()
        run_demo(self.cfg, progress_callback)
        self.enable_or_disable_all_buttons()

    def reset_demo_progress_bar(self):
        """
        Resets the demo_progressBar to the default values
        """
        self.demo_progressBar.setStyleSheet('')
        self.demo_progressBar.setValue(0)

    def set_demo_progress_bar_finished(self):
        """
        Changes progress bar colour to indicate that demo was successfully finisehd
        """
        self.demo_progressBar.setStyleSheet(FINISHED_PROGRESS_BAR_STYLE_SHEET)
        self.demo_progressBar.setAlignment(QtCore.Qt.AlignCenter)

    def update_demo_progress_bar(self, progress_percent):
        """
        Updates the progress bar according to progress_percent
        :param progress_percent: (float) [0,1] the progress of the process bar
        """
        self.demo_progressBar.setValue(round(progress_percent*100))

    ######### Functionalities for preprocessing

    def preprocess_btn_create_empty_annotation_files_clicked(self):
        """
        Creates the required directories and annotation files
        :return:
        """
        confirmed = showDialog("Create empty folder structure",
                               "Please use this only to create the initial folder structure. Otherwise your prepared data will be overwritten")
        if confirmed:
            worker = Worker(self.create_folder_structure_from_worker)
            self.threadpool.start(worker)

    def create_folder_structure_from_worker(self, progress_callback):
        """
        Calls the create folder structure function from a worker
        """
        self.enable_or_disable_all_buttons()
        create_folder_structure(self.cfg, progress_callback)
        self.enable_or_disable_all_buttons()

    def preprocess_btn_extract_frames_from_videos_and_create_framelist_files_clicked(self):
        """
        Extracts the frames from the video data
        :return:
        """
        confirmed = showDialog("Extract frames from videos",
                               "Please use this function only once or if something went wrong in the first try.")
        if confirmed:
            worker = Worker(self.extract_frames_from_videos_and_create_framelist_file_from_worker)
            self.threadpool.start(worker)

    def extract_frames_from_videos_and_create_framelist_file_from_worker(self, progress_callback):
        """
        Extracts frames from the gt videos and creates framelist files from worker
        :return:
        """
        self.enable_or_disable_all_buttons()
        extract_frames_from_videos_and_create_framelist_files(self.cfg, progress_callback)
        self.enable_or_disable_all_buttons()

    def preprocess_btn_compute_train_predict_box_list_clicked(self):
        """
        Creates the train predict box lists
        :return:
        """
        confirmed = showDialog("Predict additional boxes for training",
                               "Please use this function only once or if something went wrong in the first try.")
        if confirmed:
            worker = Worker(self.compute_train_predict_box_list_and_create_file_from_worker)
            self.threadpool.start(worker)

    def compute_train_predict_box_list_and_create_file_from_worker(self, progress_callback):
        """
        Creates the train predict box lists from worker
        :param progress_callback:
        :return:
        """
        self.enable_or_disable_all_buttons()
        compute_train_predict_box_list_and_create_file(self.cfg, visualize_results=False, progress_callback=progress_callback)
        self.enable_or_disable_all_buttons()

    def preprocess_btn_compute_test_predict_boxes_clicked(self):
        """
        Predicts bounding boxes for test from worker
        :return:
        """
        confirmed = showDialog("Predict bounding boxes for test",
                               "Please use this function only once or if something went wrong in the first try.")
        if confirmed:
            worker = Worker(self.compute_test_predict_boxes_and_create_file_from_worker)
            self.threadpool.start(worker)


    def compute_test_predict_boxes_and_create_file_from_worker(self, progress_callback):
        """
        Predicts bounding boxes for test from worker
        :param progress_callback:
        :return:
        """
        self.enable_or_disable_all_buttons()
        compute_test_predict_boxes_and_create_file(self.cfg, progress_callback)
        self.enable_or_disable_all_buttons()


    def preprocess_btn_compute_rgb_mean_and_std_clicked(self):
        """
        computes the mean and the standard deviation
        :return:
        """
        confirmed = showDialog("Compute RGB mean and std",
                               "Please check the DATASET.md on when to use this feature")
        if confirmed:
            worker = Worker(self.compute_mean_and_std_from_worker)
            self.threadpool.start(worker)

    def compute_mean_and_std_from_worker(self, progress_callback):
        """
        Computes the mean and the standard deviation from worker
        :param progress_callback:
        :return:
        """
        self.enable_or_disable_all_buttons()
        compute_mean_and_std(self.cfg, progress_callback)
        self.enable_or_disable_all_buttons()

    ######### Functionalities for train

    def train_btn_start_train_clicked(self):
        """
        Starts the training of a model
        :return:
        """
        confirmed = showDialog("Start training",
                               "Start the training if you have checked your settings")
        if confirmed:
            worker = Worker(self.launch_training_from_worker)
            self.threadpool.start(worker)


    def launch_training_from_worker(self, progress_callback):
        """
        Starts the training of a model from a worker
        :param progress_callback:
        :return:
        """
        self.enable_or_disable_all_buttons()
        launch_job(cfg=self.cfg, init_method="tcp://localhost:9999", func=train)
        self.enable_or_disable_all_buttons()

    ######### Functionalities for test

    def test_btn_start_test_clicked(self):
        """
        Starts the test of a model
        :return:
        """
        confirmed = showDialog("Start test",
                               "Start the test if you have checked your settings")
        if confirmed:
            worker = Worker(self.launch_test_from_worker)
            self.threadpool.start(worker)


    def launch_test_from_worker(self, progress_callback):
        """
        Starts the test of a model from a worker
        :param progress_callback:
        :return:
        """
        self.enable_or_disable_all_buttons()
        launch_job(cfg=self.cfg, init_method="tcp://localhost:9999", func=test)
        self.enable_or_disable_all_buttons()

    ######### Adjust config files for current configuration tab

    def current_config_btn_select_other_cfg_clicked(self):
        """
        Let's the user pick another config file an loads the respective values to the gui
        :return:
        """
        name_filter = "yaml (*.yaml)"
        cfg_path = QFileDialog.getOpenFileName(filter=name_filter)
        # Get the filename and update values
        cfg_path_filename = cfg_path[0]
        if cfg_path_filename:
            # Adjust other values and re-initialize GUI
            set_path_to_current_cfg(cfg_path_filename)
            self.path_to_current_cfg = cfg_path_filename

            self.current_config_label_path_to_current_config_value.setText(cfg_path_filename)

            self.cfg = load_config(self.path_to_current_cfg)

            # Initialize the gui values with the config
            self.load_config_to_gui()


    def current_config_btn_select_other_dataset_folder_clicked(self):
        """
        Selects a new dataset folder and updates gui as well as new folder paths
        :return:
        """
        new_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if new_folder:
            # Update config
            self.cfg.OUTPUT_DIR = new_folder
            # Update GUI
            self.current_config_label_path_to_current_dataset_folder_value.setText(new_folder)
            
            # Update Folder variables
            self.cfg = update_folder_paths(self.cfg)

            # Update tensorboard
            self.plainTextEdit_edit_log_console.setPlainText(
                "To visualize your training and test process use the following command:\ntensorboard  --logdir " + os.path.join(
                    self.cfg.OUTPUT_DIR, "runs-custom"))

    def current_config_btn_save_changes_to_current_cfg_clicked(self):
        """
        Overwrites the current config file
        :return:
        """
        confirmed = showDialog("Save changes to existing config?","Are you sure? This will overwrite all settings for this config file.")
        if confirmed:
            # Create new yaml file
            write_yaml_file(self.cfg, self.path_to_current_cfg)


    def current_config_btn_save_changes_to_new_cfg_clicked(self):
        """
        Creates a new config file at the selected path
        :return:
        """
        name_filter = "yaml (*.yaml)"
        file_path_and_filter = QFileDialog.getSaveFileName(filter=name_filter)

        if file_path_and_filter[0]:
            file_path = file_path_and_filter[0]
            # Adjust file ending
            file_path = file_path.split(".")[0] + ".yaml"

            # Create new yaml file
            write_yaml_file(self.cfg, file_path)

            #update gui
            self.current_config_label_path_to_current_config_value.setText(file_path)
            # Adjust other values and re-initialize GUI
            set_path_to_current_cfg(file_path)
            self.path_to_current_cfg = file_path

    def current_config_spinBox_NUM_GPUS_value_changed(self):
        """

        :return:
        """
        self.cfg.NUM_GPUS = self.current_config_spinBox_NUM_GPUS.value()

        # Update the batch_sizes for training and test
        self.update_train_batch_size(self.cfg.TRAIN.BATCH_SIZE)
        self.update_test_batch_size()

    def current_config_spinBox_NUM_WORKERS_value_changed(self):
        """

        :return:
        """
        self.cfg.DATA_LOADER.NUM_WORKERS = self.current_config_spinBox_NUM_WORKERS.value()

    ######### Adjust config files for custom dataset tab

    def custom_dataset_checkBox_bgr_state_changed(self):
        """

        :return:
        """
        self.cfg.CUSTOM_DATASET.BGR = self.custom_dataset_checkBox_bgr.isChecked()
        self.cfg = adjust_activity_recognizer_settings(self.cfg)

    def custom_dataset_comboBox_image_processing_backend_current_index_changed(self):
        """

        :return:
        """
        self.cfg.CUSTOM_DATASET.IMG_PROC_BACKEND = str(self.custom_dataset_comboBox_image_processing_backend.currentText())
        self.cfg = adjust_activity_recognizer_settings(self.cfg)

    def custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh_value_changed(self):
        """

        :return:
        """
        self.cfg.CUSTOM_DATASET.DETECTION_SCORE_THRESH = self.custom_dataset_doubleSpinBox_custom_dataset_detection_score_thresh.value()

    def custom_dataset_spinBoxd_frame_rate_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.CUSTOM_DATASET.FRAME_RATE = self.custom_dataset_spinBoxd_frame_rate.value()

    ######### Adjust config files for demo tab

    def demo_checkBox_show_video_state_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.VIDEO_SHOW_VIDEO_ENABLE = self.demo_checkBox_show_video.isChecked()

    def demo_checkBox_show_video_debugging_info_state_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.VIDEO_SHOW_VIDEO_DEBUGGING_INFO = self.demo_checkBox_show_video_debugging_info.isChecked()

    def demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH = round(self.demo_doubleSpinBox_VIDEO_ACTION_CATEGORY_DISPLAY_SCORE_TRESH.value(), 2)

    def demo_doubleSpinBox_export_action_recognition_score_tresh_value_changed(self):
        """
        Adjust the config file to changes in the gui
        :return:
        """
        self.cfg.DEMO.EXPORT_MIN_CATEGORY_EXPORT_SCORE = round(self.demo_doubleSpinBox_export_action_recognition_score_tresh.value(), 2)

    def demo_spinBox_detectron2_batch_size_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DETECTRON.DEMO_BATCH_SIZE = self.demo_spinBox_detectron2_batch_size.value()

    def demo_doubleSpinBox_detectron2_person_score_thresh_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DETECTRON.DEMO_PERSON_SCORE_THRESH = self.demo_doubleSpinBox_detectron2_person_score_thresh.value()

    def demo_btn_select_video_file_clicked(self):
        """
        The dialog for choosing the demo video file
        """
        # The extension, which we want allow to select
        general_type = "video"
        # Get the filter
        name_filter = get_QFileDialog_filter_for_general_type(general_type)
        # Only allow selecting certain files of type general_type
        video_filename = QFileDialog.getOpenFileName(filter=name_filter)
        if video_filename:
            # Get the filename and update values
            path_video_filename = video_filename[0]
            self.demo_label_video_file.setText(path_video_filename)
            self.cfg.DEMO.VIDEO_SOURCE_PATH = path_video_filename

    def demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE_state_changed(self):
        self.cfg.DEMO.VIDEO_EXPORT_VIDEO_ENABLE = self.demo_checkBox_VIDEO_EXPORT_VIDEO_ENABLE.isChecked()

    def demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR_value_changed(self):
        self.cfg.DEMO.VIDEO_DISPLAY_SCALING_FACTOR = self.demo_doubleSpinBox_VIDEO_DISPLAY_SCALING_FACTOR.value()

    def demo_checkBox_EXPORT_EXPORT_RESULTS_state_changed(self):
        self.cfg.DEMO.EXPORT_EXPORT_RESULTS =  self.demo_checkBox_EXPORT_EXPORT_RESULTS.isChecked()

    def demo_checkBox_MULTIPLE_ACTION_POSSIBLE_state_changed(self):
        self.cfg.CUSTOM_DATASET.MULTIPLE_ACTION_POSSIBLE = self.demo_checkBox_MULTIPLE_ACTION_POSSIBLE.isChecked()

    def demo_spinBox_DEMO_MIN_BOX_HEIGHT_value_changed(self):
        self.cfg.DETECTRON.DEMO_MIN_BOX_HEIGHT = self.demo_spinBox_DEMO_MIN_BOX_HEIGHT.value()


    ######### Adjust config files for Activity Recognizer tab

    def action_recognizer_line_edit_checkpoint_file_path_text_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.ACTIONRECOGNIZER.CHECKPOINT_FILE_PATH = self.action_recognizer_line_edit_checkpoint_file_path.text()

    def action_recognizer_comboBox_checkpoint_type_current_index_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.ACTIONRECOGNIZER.CHECKPOINT_TYPE = str(self.action_recognizer_comboBox_checkpoint_type.currentText())

    def action_recognizer_spinBox_NUM_CLASSES_value_changed(self):
        """

        :return:
        """
        self.cfg.MODEL.NUM_CLASSES = self.action_recognizer_spinBox_NUM_CLASSES.value()

    ######### Adjust config files for detectron2 tab

    def detectron2_line_edit_detection_model_cfg_text_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DETECTRON.DETECTION_MODEL_CFG = self.detectron2_line_edit_detection_model_cfg.text()

    def detectron2_line_edit_model_weights_text_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DETECTRON.MODEL_WEIGHTS = self.detectron2_line_edit_model_weights.text()

    ######### Adjust config files for deepsort tab

    def deep_sort_line_edit_reid_ckpt_text_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_REID_CKPT = self.deep_sort_line_edit_reid_ckpt.text()

    def deep_sort_doubleSpinBoxd_max_dist_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_MAX_DIST = round(self.deep_sort_doubleSpinBoxd_max_dist.value(), 2)

    def deep_sort_doubleSpinBoxd_min_confidence_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_MIN_CONFIDENCE = round(self.deep_sort_doubleSpinBoxd_min_confidence.value(), 2)

    def deep_sort_doubleSpinBoxd_nms_max_overlap_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_NMS_MAX_OVERLAP = round(self.deep_sort_doubleSpinBoxd_nms_max_overlap.value(), 2)

    def deep_sort_doubleSpinBoxd_max_iou_distance_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_MAX_IOU_DISTANCE = round(self.deep_sort_doubleSpinBoxd_max_iou_distance.value(), 2)

    def deep_sort_spinBoxd_max_age_value_changed(self):
        """
        Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_MAX_AGE = self.deep_sort_spinBoxd_max_age.value()

    def deep_sort_spinBoxd_n_init_value_changed(self):
        """
         Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_N_INIT = self.deep_sort_spinBoxd_n_init.value()

    def deep_sort_spinBoxd_nn_budget_value_changed(self):
        """
         Adjust the config file to changes in the gui
        """
        self.cfg.DEMO.DEEPSORT_NN_BUDGET = self.deep_sort_spinBoxd_nn_budget.value()

    ######### Adjust config files for train tab

    def train_spinBox_EVAL_PERIOD_value_changed(self):

        self.cfg.TRAIN.EVAL_PERIOD = self.train_spinBox_EVAL_PERIOD.value()

    def train_spinBox_CHECKPOINT_PERIOD_value_changed(self):

        self.cfg.TRAIN.CHECKPOINT_PERIOD = self.train_spinBox_CHECKPOINT_PERIOD.value()

    def train_spinBox_BATCH_SIZE_value_changed(self):

        self.update_train_batch_size(self.train_spinBox_BATCH_SIZE.value())

    def update_train_batch_size(self, proposed_value):
        """
        Train batch size has to be a multiple of cfg.NUM_GPUS
        :param proposed_value: the newly proposed value
        :return:
        """
        new_value = proposed_value

        if proposed_value % self.cfg.NUM_GPUS != 0:
            new_value = self.cfg.NUM_GPUS

        # Update cfg
        self.cfg.TRAIN.BATCH_SIZE = new_value

        # Update GUI
        self.train_spinBox_BATCH_SIZE.setValue(new_value)
        self.train_spinBox_BATCH_SIZE.setSingleStep(self.cfg.NUM_GPUS)


    def train_line_edit_CHECKPOINT_FILE_PATH_text_changed(self):

        self.cfg.TRAIN.CHECKPOINT_FILE_PATH = self.train_line_edit_CHECKPOINT_FILE_PATH.text()

    def test_ceckbox_force_AUTO_RESUME_state_changed(self):

        self.cfg.TRAIN.AUTO_RESUME = self.test_ceckbox_force_AUTO_RESUME.isChecked()

    def train_comboBox_CHECKPOINT_TYPE_current_index_changed(self):

        self.cfg.TRAIN.CHECKPOINT_TYPE = self.train_comboBox_CHECKPOINT_TYPE.currentText()

    def train_ceckbox_FINETUNE_state_changed(self):

        self.cfg.TRAIN.FINETUNE = self.train_ceckbox_FINETUNE.isChecked()

    def train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE_state_changed(self):

        self.cfg.TRAIN.FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE = self.train_ceckbox_FINETUNE_FORCE_TRAIN_EPOCH_TO_ONE.isChecked()

    def train_comboBox_HEAD_ACT_current_index_changed(self):

        self.cfg.MODEL.HEAD_ACT =  self.train_comboBox_HEAD_ACT.currentText()

    def train_spinBox_MAX_EPOCH_value_changed(self):

        self.cfg.SOLVER.MAX_EPOCH = self.train_spinBox_MAX_EPOCH.value()

    def train_doubleSpinBox_WARMUP_EPOCHS_value_changed(self):

        self.cfg.SOLVER.WARMUP_EPOCHS = self.train_doubleSpinBox_WARMUP_EPOCHS.value()

    def train_comboBox_LOSS_FUNC_current_index_changed(self):

        self.cfg.MODEL.LOSS_FUNC = self.train_comboBox_LOSS_FUNC.currentText()

    def train_doubleSpinBoxd_BASE_LR_value_changed(self):

        self.cfg.SOLVER.BASE_LR = self.train_doubleSpinBoxd_BASE_LR.value()

    ######### Adjust config files for test tab

    def test_line_edit_CHECKPOINT_FILE_PATH_text_changed(self):
        self.cfg.TEST.CHECKPOINT_FILE_PATH =  self.test_line_edit_CHECKPOINT_FILE_PATH.text()

    def test_comboBox_CHECKPOINT_TYPE_current_index_changed(self):
        self.cfg.TEST.CHECKPOINT_TYPE = self.test_comboBox_CHECKPOINT_TYPE.currentText()

    def update_test_batch_size(self):
        """
        Adjusts the test batch_size to the current num_GPUs
        :return:
        """
        self.cfg.TEST.BATCH_SIZE = self.cfg.NUM_GPUS
        self.test_spinBox_BATCH_SIZE.setValue(self.cfg.TEST.BATCH_SIZE)


######### Some general helper function
def get_QFileDialog_filter_for_general_type(general_type):
    """
    Uses the two functions to create the name_filter
    :param general_type: (string) the general type for which we want the endings (e.g., video)
    :return:
        name_filter: (string) the name filter (e.g., "video (*.mp4 *.mov)"
    """

    extensions_list = get_extensions_for_type(general_type)
    return build_file_dialog_name_filter(general_type, extensions_list)

def get_extensions_for_type(general_type):
    """
    Returns potential file endings for a certain general_type
    :param general_type: (string) the general type for which we want the endings (e.g., video)
    :return:
        extensions_list: list of strings that are the file endings, example for one element: ".mp4"
    """
    mimetypes.init()
    extensions_list = []
    for ext in mimetypes.types_map:
        if mimetypes.types_map[ext].split('/')[0] == general_type:
            extensions_list.append(ext)

    return extensions_list

def build_file_dialog_name_filter(general_type, extension_file_endings):
    """
    Builds the filter for the general type based on the extension file endings
    :param general_type: (string) the general type for which we want the endings (e.g., video)
    :param extension_file_endings: list[string] list of strings that are the file endings, example for one element: ".mp4"
    :return:
        name_filter: (string) the name filter (e.g., "video (*.mp4 *.mov)"
    """
    middle_part = ""

    for file_ending in extension_file_endings:
        middle_part = middle_part + "*" + file_ending + " "

    middle_part = middle_part.rstrip()

    # The filter we will use to select the video file
    name_filter = general_type + " (" + middle_part + ")"

    return name_filter

def get_path_to_current_cfg():
    """
    Gets the path of the currently selected config file
    :return:
        path_to_current_cfg: (str) the relative path to the current config file
    """
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    path_to_current_cfg_file = os.path.join(THIS_FOLDER, 'path_to_current_cfg.txt')

    with open(path_to_current_cfg_file, 'r') as file:
        path_to_current_cfg = file.read().replace('\n', '')

    return path_to_current_cfg

def set_path_to_current_cfg(path_to_current_cfg):
    """
    Sets the path of the currently selected config file
    :param path_to_current_cfg: (str) the relative path to the current config file
    :return:
    """
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    path_to_current_cfg_file = os.path.join(THIS_FOLDER, 'path_to_current_cfg.txt')

    txt_file = open(path_to_current_cfg_file, 'w')
    txt_file.write(path_to_current_cfg)
    txt_file.close()

def load_config(path_to_current_cfg):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    cfg.merge_from_file(path_to_current_cfg)

    # Update important folder paths
    cfg = update_folder_paths(cfg)
    cfg = adjust_activity_recognizer_settings(cfg)


    return cfg

def update_folder_paths(cfg):
    """
    Uses the cfg.OUTPUT_DIR to update all important paths
    :return:
    """

    if os.path.exists(cfg.OUTPUT_DIR):
        cfg.TRAIN.DATASET = "custom"
        cfg.TEST.DATASET = "custom"

        cfg.DATA.PATH_TO_DATA_DIR = cfg.OUTPUT_DIR
        cfg.CUSTOM_DATASET.FRAME_DIR = os.path.join(cfg.OUTPUT_DIR, "frames")
        cfg.CUSTOM_DATASET.FRAME_LIST_DIR = os.path.join(cfg.OUTPUT_DIR, "frame_lists")
        cfg.CUSTOM_DATASET.ANNOTATION_DIR = os.path.join(cfg.OUTPUT_DIR, "annotations")
        cfg.CUSTOM_DATASET.DEMO_DIR = os.path.join(cfg.OUTPUT_DIR, "demo")
        cfg.DEMO.OUTPUT_FOLDER = cfg.CUSTOM_DATASET.DEMO_DIR
        cfg.PREPROCESS.ORIGINAL_VIDEO_DIR = os.path.join(cfg.OUTPUT_DIR, "videos")
        cfg.ACTIONRECOGNIZER.LABEL_MAP_FILE = os.path.join(cfg.CUSTOM_DATASET.ANNOTATION_DIR, cfg.CUSTOM_DATASET.LABEL_MAP_FILE)

    return cfg


def adjust_activity_recognizer_settings(cfg):
    """
    Synchronize the Activity Recognizer options
    :param cfg:
    :return:
    """

    cfg.ACTIONRECOGNIZER.BGR = cfg.CUSTOM_DATASET.BGR
    cfg.ACTIONRECOGNIZER.IMG_PROC_BACKEND = cfg.CUSTOM_DATASET.IMG_PROC_BACKEND

    return cfg

def write_yaml_file(cfg, path):
    """
    Writes the cfg file at the path
    :param cfg: containing the yaml information
    :param path: the path to the new yaml file
    :return:
    """
    cfg_as_string = cfg.dump()
    yaml_file = open(path, 'w')
    yaml_file.write(cfg_as_string)
    yaml_file.close()

def showDialog(window_title, message_text):
    """
    Creates a dialog, in which a user has to confirm his choice
    :param message: (str) the text of the message
    :param window_title: (str) the title of the window
    :return: 
    """
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText(message_text)
    msgBox.setWindowTitle(window_title)
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

    returnValue = msgBox.exec()
    if returnValue == QMessageBox.Ok:
        return True
    else:
        return False

######### Start the demo gui
def start_demo_gui():
    """
    Starts the GUI
    :param cfg:
    :return:
    """
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    app.exec_()



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    start_demo_gui()
