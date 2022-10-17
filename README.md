# VQGAN

Vector Quantized Generative Adversarial Network (VGGAN) achieves a performant fusion architecture of CNN and Transformer, now become an prevalent backbone network.
In this work, Let's inspect the deails of VQGAN and get some insights to reproduce Unicolor.

## Inference

```sh
# S-FLCKR
python scripts/sample_conditional.py -r logs/2020-11-09T13-31-51_sflckr/

# ImageNet
python scripts/sample_fast.py -r logs/2021-04-03T19-39-50_cin_transformer/ -n 10 -k 600 -t 1.0 -p 0.92 --batch_size 10

# FFHQ
python scripts/sample_fast.py -r logs/2021-04-23T18-19-01_ffhq_transformer/

# coco
python scripts/sample_conditional.py -r logs/2021-01-20T16-04-20_coco_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.coco.Examples}}}"
```

## Training

We need both of 1st and 2nd training codes, becuase UniColor use them for training.
The instruction for training described in original README is like below

Training on your own dataset can be beneficial to get better tokens and hence better images for your domain.
Those are the steps to follow to make this work:
1. install the repo with `conda env create -f environment.yaml`, `conda activate taming` and `pip install -e .`
1. put your .jpg files in a folder `your_folder`
2. create 2 text files a `xx_train.txt` and `xx_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
3. adapt `configs/custom_vqgan.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU. 

<span style="color:red">Check whether the train code includes both states or not</span>

### Training for 1st Stage

### Training for 2nd Stage


```sh
python main.py --base configs/coco_scene_images_transformer.yaml -t True --gpus 0
python main.py --base configs/open_images_scene_images_transformer.yaml -t True --gpus 0

```


Download first-stage models [COCO-8k-VQGAN](https://heibox.uni-heidelberg.de/f/78dea9589974474c97c1/) for COCO or [COCO/Open-Images-8k-VQGAN](https://heibox.uni-heidelberg.de/f/461d9a9f4fcf48ab84f4/) for Open Images.
Change `ckpt_path` in `data/coco_scene_images_transformer.yaml` and `data/open_images_scene_images_transformer.yaml` to point to the downloaded first-stage models.
Download the full COCO/OI datasets and adapt `data_path` in the same files, unless working with the 100 files provided for training and validation suits your needs already.

Code can be run with
`python main.py --base configs/coco_scene_images_transformer.yaml -t True --gpus 0,`
or
`python main.py --base configs/open_images_scene_images_transformer.yaml -t True --gpus 0,`

## Reproduce Memo

### Check Inference

- Let's make the inference code available <span style="color:green">(Done)</span>
- Check the pretrained encoder available <span style="color:green">(Done)</span>
  - To the best of my knowledge, They only provide pretrained encoder and decoder using COCO dataset or Open Images.
- Check the reconstruction quality
  - If the image fidelity and high frequency deital is performant enough, this is to show a high potentiality.
- Draw procedure diagram with high and low level.
  - How the transformer stocastically infers images.

### Check Train
- check the 1st stage training code
- check the 2nd stage training code
- Make the train code available, and Draw procedure diagram with high and low level.




## Issues

#### ModuleNotFoundError: No module named 'main' <span style="color:green">(Solved)</span>

Install repository itself.

```
install pip -e .
```
