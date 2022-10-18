# VQGAN

Vector Quantized Generative Adversarial Network (VGGAN) achieves a performant fusion architecture of CNN and Transformer, now become an prevalent backbone network.
In this work, Let's inspect the details of VQGAN and get some insights to reproduce Unicolor.

## Reconstruction

The author provides an [example code](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb#scrollTo=3RxdhDGtyJ4q) to enable checking the reconstruction quality. 
I move and refine the example code into a file, recon.py. 
The example uses three types of variants `VQGAN(f8,8192)`, `VGGAN(f16, 16384)`, `VQGAN(f16, 1024)`.
The notation `f{N}` means the spatial resolution of feature as K/N by K/N w.r.t. the input image size of K by K, that is `f8` and `f16` means 32 by 32 and 16 by 16 spatial resolution from 256 by 256 imput image.
The second number refers to the number of codebook entries.
Although not described, the dimensionality of codebook entries are generally 256.
A result example is as
![sampe](outputs/sample1.jpg) 
As shown the figure, the high spatial dimension, VQGAN(f8, 8192), shows the most high fidelity w.r.t. image structure.


### Input Image Preprocessing

They use input images reranged from [0, 1] to [-1, 1]. 
The code example is as below.

```python
def preprocess_vqgan(x):
  x = 2. * x - 1.
  return x
```

### Model Components

The _Gumbel_ means <span style="color:red">XX</span>.
```bash
GumbelVQ(
  (encoder): Encoder(
    (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (down): ModuleList(
      (0): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (1): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (2): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (3): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList(
          (0): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
    )
    (mid): Module(
      (block_1): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (attn_1): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (block_2): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
    (conv_out): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (decoder): Decoder(
    (conv_in): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (mid): Module(
      (block_1): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (attn_1): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (block_2): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (up): ModuleList(
      (0): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
      )
      (1): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (2): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (3): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList(
          (0): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (2): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (upsample): Upsample(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
    (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (loss): DummyLoss()
  (quantize): GumbelQuantize(
    (proj): Conv2d(256, 8192, kernel_size=(1, 1), stride=(1, 1))
    (embed): Embedding(8192, 256)
  )
  (quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (post_quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
)
```

## Random Generation

```sh
# S-FLCKR
python scripts/sample_conditional.py -r logs/2020-11-09T13-31-51_sflckr/

# ImageNet
python scripts/sample_fast.py -r logs/2021-04-03T19-39-50_cin_transformer/ -n 10 -k 600 -t 1.0 -p 0.92 --batch_size 10

# FFHQ
python scripts/sample_fast.py -r logs/2021-04-23T18-19-01_ffhq_transformer/

# COCO
CUDA_VISIBLE_DEVICES=1 streamlit run scripts/sample_conditional.py -- -r logs/2021-01-20T16-04-20_coco_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.coco.Examples}}}"
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

- Train 1st step VQGAN and achieve a proper quality <span style="color:red">(DoIt)</span>
  - Prepare my custom dataset for birds <span style="color:green">(Done)</span>
  - Find the way to see their training lo <span style="color:green">(Done)</span>g
- Train 2st step VQGAN and achieve a proper quality
- Train 1st step Chroma-VQGAN and achieve a proper quality
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
