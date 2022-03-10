# Reproduce Our Results

This document shows how the results can be reproduced.

We obtained our results using a [Data Science Virtual Machine](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-dsvm.ubuntu-1804?tab=overview). As part of our non-functional evaluation, we tested the prototype using different hardware settings.

If you follow the instructions below with the same setup, you will obtain identical results. Other settings in your hardware setup as well as in your demo configuration may lead to minor deviations.


## Step 1: Start The GUI
Please make sure you have properly installed the VideoProcessMining project following the instruction in [INSTALL.md](INSTALL.md).
- Follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md#the-basic-principle) to start the GUI.

## Step 2: Choose The Configuration File And Create The Directory Structure
- Create a new root directory for the demo (e.g., "/home/user/Desktop/crepe_demo")
- "Current Configuration" tab: Use the button "Select other config" to select the config file [CREPE_EVAL_SLOWFAST_32x2_R50_SHORT.yaml](configs/CREPE_EVAL_SLOWFAST_32x2_R50_SHORT.yaml)
- "Current Configuration" tab: Use the button "Select other dataset folder" to select to the newly created root directory
- "Current Configuration" tab: Save your settings by pressing the button "Save changes to new config" and save the configuration file in your newly created root directory for the demo
- "Preprocess" tab: Press the button "Create folder structure and empty annotation files" to create the directory structure

## Step 3: Download required files for the Crepe Dataset
- Download the [original video files](https://osf.io/d5k38/files) DSC_4081.mp4 and DSC_4083.mp4 and copy them into the "dataset_name/video" folder
- Download the [trained model](https://www.fim-rc.de/wp-content/uploads/crepe_eval_model.pyth) file that is used in the evaluation of our paper and copy it into the "dataset_name/checkpoints" folder
- Download the [label_map_file.pbtxt](slowfast/datasets/crepe_dataset/label_map_file.pbtxt) file and copy it into the "dataset_name/annotations" folder (replace the automatically inserted file). You will also find the file in your repository (slowfast/datasets/crepe_dataset/label_map_file.pbtxt).

## Step 4: Adjust Your Settings For The Demo
- "Activity Recognizer" tab, "Checkpoint file path to pretrained model": Insert the path to the previously downloaded model for event log extraction, which is placed in your "dataset_name/checkpoints" folder
- "Demo" tab: If you want the prototype to visualize its predictions, check the checkbox "Annotate and show video"
- "Demo" tab: If you want the prototype to provide a video file with all annotations after the demo, check the checkbox "Export annotated video as video file" (this will require approx. 8 GB memory)
- "Current Configuration" tab: Save your current configuration using the button "Save changes to current config"

## Step 5: Run Demo for DSC_4081.mp4
We used a [Standard_NV24_Promo](https://docs.microsoft.com/en-us/azure/virtual-machines/nv-series) virtual machine configuration with 4 GPUs.
- "Current Configuration" tab: Set the value of the field "Max number of GPUs" to 4 (if supported by your hardware)
- "Demo" tab: Use the button "Select video file" to select the video DSC_4081.mp4 in the "dataset_name/video" folder
- "Demo" tab: Start information extraction using the button "Start Demo"
- After the demo is finished, the event logs are transferred to an automatically created subfolder in the "dataset_name/demo" directory. The files ending with "log_50_thresh" (equalling an export threshold of 50 %) contain the results presented in our paper.

## Step 6: Run Demo for DSC_4083.mp4
We used a [Standard_NV6_Promo](https://docs.microsoft.com/en-us/azure/virtual-machines/nv-series) virtual machine configuration with 1 GPU.
- "Current Configuration" tab: Set the value of the field "Max number of GPUs" to 1
- "Demo" tab: Use the button "Select video file" to select the video DSC_4083.mp4 in the "dataset_name/video" folder
- "Demo" tab: Start information extraction using the button "Start Demo"
- After the demo is finished, the event logs are transferred to an automatically created subfolder in the "dataset_name/demo" directory. The files ending with "log_50_thresh" (equalling an export threshold of 50 %) contain the results presented in our paper.


Since the prototype generates separate logs for each video file, they must be merged into one event log, which corresponds to the INFERRED_LOG presented in our work.

