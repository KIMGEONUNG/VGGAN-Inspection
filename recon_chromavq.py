import sys
import requests
import io
import yaml
import torch
from omegaconf import OmegaConf
from taming.models.chromavq import ChromaVQ
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

sys.path.append(".")
font = ImageFont.truetype(
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


def preprocess(img, target_image_size=256, map_dalle=True):
  s = min(img.size)

  if s < target_image_size:
    raise ValueError(f'min dim for image {s} < {target_image_size}')

  r = target_image_size / s
  s = (round(r * img.size[1]), round(r * img.size[0]))
  img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
  img = TF.center_crop(img, output_size=2 * [target_image_size])
  img = torch.unsqueeze(T.ToTensor()(img), 0)
  return img


def postprocess(x):
  return ((x + 1) / 2).clip(0, 1)


def togray(x):
  if len(x.shape) == 3:
    x = x[None, ...]
  g = F.conv2d(x, torch.randn(1, 3, 1, 1))
  g = torch.tanh(g)
  return g


def view_img(x):
  x = postprocess(x)
  x = T.ToPILImage()(x)
  x.show()


def save_img(x, path):
  if len(x.shape) == 4:
    x = x[0]
  x = postprocess(x)
  x = T.ToPILImage()(x)
  x.save(path)


if __name__ == "__main__":

  path_ckpt = "logs/2022-11-15T12-39-40_chroma_vqgan_mark/checkpoints/last.ckpt"
  path_config = "logs/2022-11-15T12-39-40_chroma_vqgan_mark/configs/2022-11-15T12-39-40-project.yaml"

  # DISABLE GRAD TO SAVE MEMORY
  torch.set_grad_enabled(False)
  # SET DEVICE
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # LOAD DATASET
  path_image = "datasets/valid_subset_birds10/n01537544/ILSVRC2012_val_00002071.JPEG"
  img = Image.open(path_image)
  x = preprocess(img, 256)
  x = x * 2 - 1

  # LOAD MODEL
  config = load_config(path_config, display=False)
  model = ChromaVQ(**config.model.params)
  sd = torch.load(path_ckpt, map_location="cpu")["state_dict"]
  missing, unexpected = model.load_state_dict(sd, strict=False)
  model.eval().to(DEVICE)

  # RECONSTRUCTION
  for i in range(100):
    x_g = togray(x)
    y, _ = model(x.to(DEVICE), x_g.to(DEVICE))
    # y = ((y + 1) * 0.5).clip(0, 1)
    # SAVE RESULTS
    save_img(x_g, path='results_recon/%02d_gray.jpg' % i)
    save_img(y, path='results_recon/%02d_recon.jpg' % i)
