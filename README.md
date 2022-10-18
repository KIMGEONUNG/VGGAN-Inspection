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

# COCO
python scripts/sample_conditional.py -r logs/2021-01-20T16-04-20_coco_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.coco.Examples}}}"
```

## Training

We need both of 1st and 2nd training codes, becuase UniColor use them for training.

### Training for 1st Stage

1. put your .jpg files in a folder `your_folder`
2. create 2 text files a `xx_train.txt` and `xx_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
3. adapt `configs/custom_vqgan.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU. 


```sh
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,
```

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

### Train Reproduce
<span style="color:blue">
The first thing we have to do is the model training, step by step.
</span>

- Train 1st step VQGAN and achieve a proper quality <span style="color:red">(DoIt)</span>
  - Prepare my custom dataset for birds
- Train 1st step Chroma-VQGAN and achieve a proper quality
- Train 2st step VQGAN and achieve a proper quality
- Train 2st step Chroma-VQGAN and achieve a proper quality

### Check Inference Code

- Let's make the inference code available <span style="color:green">(Done)</span>
- Check the pretrained encoder available <span style="color:green">(Done)</span>
  - To the best of my knowledge, They only provide pretrained encoder and decoder using COCO dataset or Open Images.
- Check the reconstruction quality
  - If the image fidelity and high frequency deital is performant enough, this is to show a high potentiality.
- Draw procedure diagram with high and low level.
  - How the transformer stocastically infers images.

### Check Train Code
- check the 1st stage training code
- check the 2nd stage training code
- Make the train code available, and Draw procedure diagram with high and low level.


## Issues

#### omegaconf.errors.ConfigAttributeError: Missing key logger <span style="color:green">(Solved)</span>

The fundamental reason is the different package versions for _pytorch-lightening_ and _omegaconfig_.
[Here](https://github.com/CompVis/taming-transformers/issues/72#issuecomment-875757912) in issue introduced some solutions.
I tried many solution, but eventually failed with subsequent issues.
Not elegant, but working solution is as below

```sh
pip install pytorch-lightning==1.0.8 omegaconf==2.0.0
```

The error message is as

```
Global seed set to 23
Running on GPUs 0,
Working with z of shape (1, 256, 16, 16) = 65536 dimensions.
loaded pretrained LPIPS loss from taming/modules/autoencoder/lpips/vgg.pth
VQLPIPSWithDiscriminator running with hinge loss.
Traceback (most recent call last):
File "main.py", line 465, in <module>
logger_cfg = lightning_config.logger or OmegaConf.create()
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 355, in __getattr__
self._format_and_raise(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/base.py", line 231, in _format_and_raise
format_and_raise(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/_utils.py", line 900, in format_and_raise
_raise(ex, cause)
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/_utils.py", line 798, in _raise
raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 351, in __getattr__
return self._get_impl(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 442, in _get_impl
node = self._get_child(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/basecontainer.py", line 73, in _get_child
child = self._get_node(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 480, in _get_node
raise ConfigKeyError(f"Missing key {key!s}")
omegaconf.errors.ConfigAttributeError: Missing key logger
full_key: logger
object_type=dict
)))))
```

#### ModuleNotFoundError: No module named 'main' <span style="color:green">(Solved)</span>

Install repository itself.

```
install pip -e .
```
