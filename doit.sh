#!/bin/bash

# THIS IS TRAIN CODE USING OUR BIRD TEST DATASET
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,
