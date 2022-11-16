#!/bin/bash

# THIS IS TRAIN CODE USING OUR BIRD TEST DATASET
python main.py --base configs/chroma_vqgan_transformer.yaml -t True --gpus 0,
