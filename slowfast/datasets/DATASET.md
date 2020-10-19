# Dataset Preparation

Make sure you have properly installed the VideoProcessMining project following the instruction in [INSTALL.md](INSTALL.md).
The VideoProcessMining project exclusively targets video data for spatio-temporal activity recognition.
This type of video data comprises at least one actor that performs at least one activity at a time.

Therefore, the VideoProcessMining project builds on [PySlowFast's](https://github.com/facebookresearch/SlowFast) functionality for the [AVA dataset](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md#ava).
The VideoProcessMining project's generic implementation allows to train activity recognition models on any custom spatio-temporal activity recognition dataset.

As foundation for all further functionality, users have to prepare the annotations for their custom dataset.

The prototype supports and automates data preparation.
Please complete every step before proceeding to the next one.
Be aware that some automated steps of the preprocessing require some minutes/hours depending on your dataset's size.
Repeat a step only if something went wrong (it is possible that you may have to delete some files), because your data will be overwritten.

## The Basic Principle

To train the activity recognition model, annotations at one-second intervals have to pe prepared.

For every full video second, users have to label the location as well as every activity of every actor.
In contrast to other approaches, we do not use annotations on a frame level.

Please follow these steps to prepare your dataset.

First start the prototype GUI (e.g., using the PyCharm IDE Python Console) with the following command:

It is possible to start the GUI from command line
```
export PYTHONPATH=/path/to/VideoProcessMining
cd /path/to/VideoProcessMining
python tools/start_gui.py
```

or an IDE. Use PyCharm's "Python Console" with the following command:
```
%run /path/to/VideoProcessMining/tools/start_gui.py
```


## Step 1: Create The Directory Structure

- Create a root directory for your custom dataset (e.g., /home/user/Desktop/dataset")
- Use the button "Select other dataset folder" in the tab "Current Configuration" to navigate to the newly created directory
- Save your settings by pressing the button "Save changes to current config"
- Press the button "Create folder structure and empty annotation files" in the "Preprocess" tab to create the directory structure

The prototype creates the required directory and inserts some empty files.

The directory structure of your dataset is as follows:

```
dataset_name
|_ annotations (contains all relevant annotation files)
    |_label_map_file.pbtxt (assigns names and ids to all activity classes)
    |_train_groundtruth.csv (contains all activity annotations for training data)
    |_train_predict_box_list.csv (contains additional bounding boxes as training input that have a high IoU with the train_groundtruth bounding boxes)
    |_val_excluded_timestamps.csv (we do not use this functionality, always empty)
    |_val_groundtruth.csv (contains all activity annotations for test data)
    |_val_predicted_boxes.csv (contains the predicted bounding boxes for testing the activity recognition model)
|_ checkpoints (folder for model checkpoints that are inserted after a predefined number of epochs)
|_ demo (contains the output event logs)
|_ frame_lists (matches annotations with the frames in the "frames" folder)
|_ frames (includes a separate folder for each video file of the "videos" folder. Each subfolder contains all frames of a specific video file)
|_ runs-custom (contains tensorboard-information)
|_ videos (contains all video files of a custom dataset)
|_ stdout.log (a file containing log information, which is added during training)
```

## Step 2: Add Your Video Files (Manually)

Please copy all your video files into the "videos" folder created in step 1.
Make sure that the filenames do not contain any spaces.

## Step 3: Prepare Initial Annotation Files (Manually)

This is the most important and time-consuming part of the annotation preparation process.
You have to label the video data by preparing all three annotation files listed in the following table.


| file_name | purpose |
| ------------- | ------------- |
| [label_map_file.pbtxt](custom_dataset/exemplary_annotation_files/label_map_file.pbtxt) | specifies all classes for the activity recognition model |
| [train_groundtruth.csv](custom_dataset/exemplary_annotation_files/train_groundtruth.csv) | contains all activity annotations for training data |
| [val_groundtruth.csv](custom_dataset/exemplary_annotation_files/val_groundtruth.csv) | contains all activity annotations for test data |

For each file, we provide an exemplary file that defines the required format.
Make sure that all files are without header.

Please insert the three files into the "annotations" folder created in step 1.


## Step 4: Finish Annotation Preparation (Mostly automated)

Now that the most important annotation information was prepared, the prototype mostly automates the remaining steps.

First, go to the tab "Custom Dataset" and check if the default settings are valid for your custom dataset.
Usually, it is not necessary to change anything.

Note that the "Targeted Frame Rate" does not have to match with your video files' frame rate.
Your video files will be automatically adjusted to your selected frame rate, which enables comparable results for activity recognition.
The default 30 fps are proposed in the original [SlowFast-Paper](https://arxiv.org/pdf/1812.03982.pdf).


### Step 4.1: Extract Frames From Videos And Create Framelist Files

Use the button "Extract frames from videos and create framelist files" in the tab "Preprocess".
The prototype automatically extracts the frames into the folder "frames".
It also inserts two files comprising all extracted frames into the folder "frame_lists".

### Step 4.2: Predict Additional Bounding Boxes For Training (Optional)

If you want to compute additional bounding boxes as training input, which have a high IoU with the GT bounding boxes, press the button "Predict additional bounding boxes for training".
It creates the file "train_predict_box_list.csv" in the folder "annotations".

Please make the following adjustments to the file:
- Remove all bounding boxes with IoU-value below your selected threshold.
- Remove the header.

### Step 4.3: Predict Bounding Boxes For Test

To jointly test your activtiy recognition model as well as the object detector, bounding boxes for test data are required.

Use the button "Predict bounding boxes for test" to automatically create the file "val_predicted_boxes.csv" in the annotations folder

### Step 4.4: Compute RGB mean and std of your dataset

If you want to train your activity recognition model from scratch, it may be beneficial to computer RGB mean and std of your training dataset.

The prototype supports this functionality: button "Compute RGB mean and std".

It creates two csv-files "data_mean.csv" and "data_std.csv" in the "annotations" directory.

Please add the computed values (as numbers) to your config file:

| file_name | config_attribute |
| ------------- | ------------- |
| data_mean.csv | DATA.MEAN |
| data_std.csv | DATA.STD |


