#!/bin/bash

# THIS IS TRAIN CODE USING OUR BIRD TEST DATASET
python main.py --base configs/chroma_vqgan.yaml -t True --accelerator gpu --devices 0,
