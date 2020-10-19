#!/usr/bin/env python3
# Created in the VideoProcessMining project

from app_gui.demo_window import start_demo_gui
import torch

def main():
    """
    Main function to spawn the train and test process.
    """

    start_demo_gui()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
