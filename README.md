# VideoProcessMining

The VideoProcessMining project provides a general approach to extract process-mining-conform event logs from unstructured video data.
It targets custom datasets and spatio-temporal activity recognition.


![](demo_VideoProcessMining/video_process_mining_demo.gif)

## Dependencies

The VideoProcessMining project uses several open source computer vision projects.

- [Detectron2](https://github.com/facebookresearch/detectron2) ([License](https://github.com/facebookresearch/detectron2/blob/master/LICENSE)): For object detection
- [Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch) ([License](https://github.com/ZQPei/deep_sort_pytorch/blob/master/LICENSE)): For object tracking
- [PySlowFast](https://github.com/facebookresearch/SlowFast) ([License](https://github.com/facebookresearch/SlowFast/blob/master/LICENSE)): For activity recognition
- [PM4Py](https://github.com/pm4py/pm4py-core) ([License](https://github.com/pm4py/pm4py-core/blob/release/LICENSE)): For XES event log generation
- [Crepe Dataset](https://osf.io/d5k38/) from the paper [STARE: Spatio-Temporal Attention Relocation for Multiple Structured Activities Detection](https://ieeexplore.ieee.org/document/7293663)
- [imutils](https://github.com/jrosebr1/imutils) ([License](https://github.com/jrosebr1/imutils/blob/master/LICENSE.txt)) for fast video frame reading

## License

The VideoProcessMining project is released under the [GNU General Public License v3.0](LICENSE).

We stated our changes in the modified files and included the original licenses in the [licenses directory](licenses).

## Functionality

The VideoProcessMining project comprises a GUI that automates and facilitates several tasks:
- Preprocessing of custom video datasets for spatio-temporal activity recognition
- Training and testing of activity recognition models
- Extraction of process-mining-conform event logs from unstructured video data (demo)

## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md).

## Dataset Preparation

You may follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare your custom dataset.

## Use VideoProcessMining For Training Or Testing Models

After preparing your dataset, you can follow the instructions in [INSTRUCTIONS.md](INSTRUCTIONS.md) to train and test your models.

You can also extract information from video data.

## Reproduce our results

If you are interested in reproducing our results, follow the instructions in [CREPE_DEMO.md](CREPE_DEMO.md).
