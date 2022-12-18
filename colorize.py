#!/usr/bin/env python

import gradio as gr
import cv2.cv2 as cv
import argparse
import importlib
import torch
from glob import glob
from os.path import join, exists
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from functools import partial
import numpy as np
import pycomar.images


def parse():
  p = argparse.ArgumentParser()
  p.add_argument("--path_log",
                 type=str,
                 default="logs/2022-11-22T23-42-43_chroma_vqgan_transformer")
  p.add_argument("--share_link", action='store_true')
  p.add_argument("--path_input",
                 type=str,
                 default="inputs/ILSVRC2012_val_00002071_resize_256.JPEG")
  return p.parse_args()


def get_obj_from_str(string, reload=False):
  module, cls = string.rsplit(".", 1)
  if reload:
    module_imp = importlib.import_module(module)
    importlib.reload(module_imp)
  return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
  if "target" not in config:
    raise KeyError("Expected key `target` to instantiate.")
  return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
  model = instantiate_from_config(config)
  if sd is not None:
    model.load_state_dict(sd)
  if gpu:
    model.cuda()
  if eval_mode:
    model.eval()
  return {"model": model}


def load_model(config, ckpt, gpu, eval_mode):
  # load the specified checkpoint
  if ckpt:
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    print(f"loaded model from global step {global_step}.")
  else:
    pl_sd = {"state_dict": None}
    global_step = None
  model = load_model_from_config(config.model,
                                 pl_sd["state_dict"],
                                 gpu=gpu,
                                 eval_mode=eval_mode)["model"]
  return model, global_step


def preprocess(img, target_image_size=256, map_dalle=True):
  # s = min(img.size)
  #
  # if s < target_image_size:
  #   raise ValueError(f'min dim for image {s} < {target_image_size}')

  # r = target_image_size / s
  # s = (round(r * img.size[1]), round(r * img.size[0]))
  # img = TF.resize(img, s, interpolation=Image.LANCZOS)
  # img = TF.center_crop(img, output_size=2 * [target_image_size])
  img = torch.unsqueeze(T.ToTensor()(img), 0)
  return img


def extract_mask(origin, img):
  mask = (origin - img) != 0
  img = mask * img
  img = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_NEAREST)
  return img


def extract_hint(img):
  img = cv.resize(img, dsize=(16, 16), interpolation=cv.INTER_NEAREST)
  return img


def extract_hint4view(img):
  img = cv.resize(img, dsize=(16, 16), interpolation=cv.INTER_NEAREST)
  img = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_NEAREST)
  return img


def togray(x):
  g = x.astype('float') / 255
  g = 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]
  g = np.tile(g[..., None], (1, 1, 3))
  return g


def togray_random(x):
  g = x.astype('float') / 255
  w = np.random.randn(3)
  g = g * 2 - 1
  g = (g @ w)[..., None]
  g = np.tanh(g).astype(np.float32)
  g = (g + 1) / 2
  g = np.tile(g, (1, 1, 3))
  return g


def prediction(x, x_g, hint, model, dev="cuda:0"):
  """
  Arguments
  ----------------------------------------------------------------------------
  x: uint8 numpy array with shape of (256, 256, 3) and the range of [0, 255]
  x_g: uint8 numpy array with shape of (256, 256, 3) and the range of [0, 255]
    all channels has same values because it represents grayscale image
  hint: uint8 numpy array with shape of (16, 16, 3) and the range of [0, 255]
    we consider the value [0, 0, 0] is the masked pixel. However, we have to
    change this state because it is possible to use black color hint.

  Returns
  ----------------------------------------------------------------------------
  Colorized results
  """
  # Preprocessing
  # Make mask (hint is -1, unknown is 0)
  mask = (hint.sum(-1) != 0).astype('uint8') * -1
  mask = mask[..., None]

  # reduce gray channel
  x_g = x_g[:, :, :1]

  # Range to [-1, 1]
  fn_rerange = lambda a: (a.astype('float32') / 255) * 2 - 1
  x = fn_rerange(x)
  x_g = fn_rerange(x_g)
  hint = fn_rerange(hint)

  # To torch tensor with transpose
  pycomar.images.torchimg2cvimg
  fn_2torch = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None, ...]
  x = fn_2torch(x)
  x_g = fn_2torch(x_g)
  hint = fn_2torch(hint)
  mask = fn_2torch(mask)

  x = x.to(device=dev)
  x_g = x_g.to(device=dev)
  mask = mask.to(device=dev)
  hint = hint.to(device=dev)

  quant_z, z_indices = model.encode_to_z(x)

  feat_g_origin = model.first_stage_model.encoder_gray(x_g)
  feat_g = feat_g_origin.view(*feat_g_origin.shape[:-2], -1)  # flatten
  feat_g = feat_g.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

  mask = mask.view(*mask.shape[:-2], -1)  # flatten
  mask = mask.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

  hint = hint.view(*hint.shape[:-2], -1)  # flatten
  hint = hint.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

  # sample
  # shape of (4, 0), Note the dim of the second is zero
  z_start_indices = z_indices[:, :0]
  index_sample = model.sample(z_start_indices,
                              feat_g,
                              mask,
                              hint,
                              steps=z_indices.shape[1],
                              temperature=1,
                              sample=True,
                              top_k=100,
                              callback=lambda k: None)

  sample = model.decode_to_img(index_sample, quant_z.shape, feat_g_origin)
  sample = ((sample[0] + 1) * 0.5).clip(0, 1)
  sample = T.ToPILImage()(sample)
  return sample


if __name__ == "__main__":
  args = parse()

  is_gui = False
  path_ckpt = glob(join(args.path_log, "checkpoints/0352.ckpt"))[0]
  path_config = glob(join(args.path_log, "configs/*-project.yaml"))[0]
  assert exists(path_ckpt)
  assert exists(path_config)
  assert exists(args.path_input)
  config = OmegaConf.load(path_config)

  # LOAD MODEL
  model, global_step = load_model(config, path_ckpt, gpu=True, eval_mode=True)
  model.eval()
  prediction_wrap = partial(prediction, model=model)
  print('Model was loaded')

  # GUI ######################################################################
  with gr.Blocks() as demo:
    gr.Markdown("# Colorization Demo")
    ## ENROLL UI COMPONENTS
    with gr.Box():
      with gr.Row():
        with gr.Column():
          image_original = gr.Image(label="Original").style(height=300)
          ex = gr.Examples(examples=[
              "inputs/ILSVRC2012_val_00002071_resize_256.JPEG",
              "inputs/ILSVRC2012_val_00005567_resize_256_256.JPEG",
              "inputs/ILSVRC2012_val_00005848_resize_256_256.JPEG",
              "inputs/ILSVRC2012_val_00045880_ccrop_0375_resize_256.JPEG",
          ],
                           inputs=image_original)
        with gr.Column():
          viewer_gray = gr.Image(interactive=False,
                                 label="Gray",
                                 shape=(256, 256)).style(height=400)
          with gr.Row():
            btn_gray = gr.Button("To Gray")
            btn_gray_random = gr.Button("To Random Gray")

    with gr.Box():
      with gr.Row():
        image_canvas = gr.Image(source="upload",
                                tool="color-sketch",
                                label="Canvas",
                                interactive=True)
        viewer_mask = gr.Image(interactive=False,
                               label="Mask",
                               shape=(256, 256)).style(height=400)
        viewer_hint = gr.Image(interactive=False,
                               label="Hint",
                               visible=False,
                               shape=(16, 16)).style(height=400)
        viewer_hint4view = gr.Image(interactive=False,
                                    label="Hint View",
                                    shape=(16, 16)).style(height=400)
    with gr.Box():
      with gr.Column():
        viewer_result = gr.Image(interactive=False,
                                 shape=(256, 256),
                                 label="Result").style(height=400)
        btn_colorize = gr.Button("Colorize!")

    ## ENROLL EVENTS
    image_original.change(lambda x: x,
                          inputs=image_original,
                          outputs=image_canvas)
    image_original.change(fn=togray,
                          inputs=image_original,
                          outputs=viewer_gray)

    btn_gray.click(fn=togray, inputs=image_original, outputs=viewer_gray)
    btn_gray_random.click(fn=togray_random,
                          inputs=image_original,
                          outputs=viewer_gray)
    image_canvas.change(fn=extract_mask,
                        inputs=[image_original, image_canvas],
                        outputs=viewer_mask)
    viewer_mask.change(fn=extract_hint,
                       inputs=viewer_mask,
                       outputs=viewer_hint)
    viewer_mask.change(fn=extract_hint4view,
                       inputs=viewer_mask,
                       outputs=viewer_hint4view)
    btn_colorize.click(fn=prediction_wrap,
                       inputs=[image_original, viewer_gray, viewer_hint],
                       outputs=viewer_result)

  demo.launch(share=args.share_link)
  ############################################################################
  print('Program finished')
