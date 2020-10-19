# Instructions for VideoProcessMining

This document provides a brief intro of using the VideoProcessMining GUI, which covers most functionality and facilitates your work.

Be aware that there are advanced configuration settings that can be modified. They are explained in [the config files](slowfast/config).

We provide two exemplary config files that should facilitate your initial steps. For other types of models see the [SlowFast Model Zoo](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md).
| Config file for | Corresponding base model for transfer learning |
| ------------- | ------------- |
|[SLOWFAST Model](configs/1_EXAMPLE_SLOWFAST_32x2_R50_SHORT.yaml)|[SLOWFAST_8x8_R50.pkl](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)|
|[SLOW Model](configs/2_EXAMPLE_C2D_8x8_R50_SHORT.yaml)|[C2D_8x8_R50.pkl](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/C2D_8x8_R50.pkl)|

Some functionality of the prototype may need some time.
If you are interested in the current progress, just look into the "stdout.log" file, which is created in your dataset directory.

## Start the GUI
Make sure you have properly installed the VideoProcessMining project following the instruction in [INSTALL.md](INSTALL.md).

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

## Train Your Model

Make sure you have prepared your dataset following [DATASET.md](slowfast/datasets/DATASET.md).

"Current Configuration" tab
- Select one of the two exemplary configs
- Check if the path to the current dataset folder is correct
- Set your desired number of GPUs for training as well as the number of workers for the dataloader.

"Activity Recognizer" tab
- Enter the number of activity classes, which corresponds to the classes defined in your "label_map_file.pbtxt"

"Train" tab
- Modify the options as desired
- If your activity classes are mutually exclusive choose "softmax" as head and "cross_entropy" as loss function. Otherwise, if your activity classes are not mutually exclusive choose "sigmoid" as head and "bce" as loss function.
- If you want to use transfer learning instead of training your model from scratch, download the respective base model (see "Corresponding base model for transfer learning" in table at the top). Copy the path to the base model into the field "Use the following model as base model for training". If you want to train your model from scratch, do not modify this field.

Save your current configuration using one of the buttons in the tab "Current Configuration".

Start training using button "Start Training"

The training may need some time. Use tensorboard to monitor your performance.

Checkpoints are automatically stored in your dataset's "checkpoints" directory.
Make sure that you copy the models to another directory before training your next model. Otherwise the checkpoints will be overwritten.

## Test your model

After having trained your model for several epochs, you can test it on your test data.

"Current Configuration" tab
- Set your desired number of GPUs for testing as well as the number of workers for the dataloader.

"Activity Recognizer" tab
- Check the number of activity classes, which corresponds to the classes defined in your "label_map_file.pbtxt"

"Test" tab
- Insert the path to the desired base model (usually a model in the "checkpoints" directory). If no path is inserted, the prototype uses the model with the highest epoch in the "checkpoints" folder. If there are no models in the "checkpoints" folder, the prototype selects the base model from training.

Save your current configuration using one of the buttons in the tab "Current Configuration".

Start testing using button "Start Test"

## Demo (Export Event Log)

"Current Configuration" tab
- Set your desired number of GPUs for the demo.

"Activity Recognizer" tab
- Insert the path to the desired model for event log extraction (usually one of the models resulting from training)
- Check the number of activity classes, which corresponds to the classes defined in your "label_map_file.pbtxt"

"Detectron2" tab
- Optional: Choose another model from the repository

"Deep Sort" tab
- Optional: Change some settings.
- Since the project only uses one checkpoint for REID, do not modify the checkpoint

"Demo" tab
- Modify the options as you prefer
- Since this is important for the event log generation, check or uncheck "An actor can perform several activities concurrently". A checked box means that the activity classes are not mutually exclusive
- Select your demo video file, from which the information is extracted (button "Select video file")

Save your current configuration using one of the buttons in the tab "Current Configuration".

Start information extraction using the button "Start Demo"

After the demo is finished, the event logs are transferred to the "demo" directory.