#!/bin/bash

python main.py --base configs/custom_vqgan.mark11.yaml -t True --gpus 0,1,2,3
