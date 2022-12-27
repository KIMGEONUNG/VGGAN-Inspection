#!/usr/bin/env python

import gradio as gr
import torch
from glob import glob
from os.path import join, exists
from omegaconf import OmegaConf
import numpy as np
from torchvision.transforms import (ToPILImage, Grayscale, Compose, ToTensor,
                                    Resize, CenterCrop)

from taming.util import load_model_from_config
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from taming.algorithms import SignalProcessor
from tqdm import tqdm

import yaml
from taming.models.vqgan import VQModel, GumbelVQ


class System(object):
    """Recolorization GUI System"""

    def __init__(self, share=True):
        self.share = share
        self.height = 300
        self.model = None
        self.hint = None

    def _load_model(self, config, ckpt, gpu, eval_mode):
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

    def set_model(self, key):

        MAP_PATH = {
            "Baseline": "logs_mark/2022-12-18T07-06-50_chroma_vqgan",
            "RandGray": "logs_mark/2022-12-18T07-08-41_chroma_vqgan_randgray",
            "VQHint": "logs_mark/2022-12-20T02-00-03_chroma_vqgan",
            "VQHint+RandGray":
            "logs_mark/2022-12-20T02-01-42_chroma_vqgan_randgray",
            "ScaleGray": "logs_mark/2022-12-21T12-25-22_chroma_vqgan",
        }

        path_log = MAP_PATH[key]

        path_ckpt = sorted(glob(join(path_log, "checkpoints/*.ckpt")))[-1]
        path_config = glob(join(path_log, "configs/*-project.yaml"))[0]
        assert exists(path_ckpt)
        assert exists(path_config)
        config = OmegaConf.load(path_config)

        model, _ = self._load_model(config,
                                    path_ckpt,
                                    gpu=True,
                                    eval_mode=True)
        self.model = model

        self.n_embed = config["model"]["params"]["n_embed"]
        self.cmap = plt.cm.get_cmap('hsv', self.n_embed)

        return model

    @torch.no_grad()
    def get_visible_cods(self, x):
        h = self.model.encoder(x)
        h = self.model.quant_conv(h)  # After that we have to use mask and hint
        quant, emb_loss, info = self.model.quantize(h)
        code = info[2].cpu().numpy()
        code_log = str(code.reshape(16, 16))
        code = self.cmap(code)[..., :-1]
        code = code.reshape(16, 16, -1)
        code = (code * 255).astype('uint8')
        return code, code_log

    def predict(self, img: np.ndarray):
        """
        img : range from 0 to 255 with unit8 and dimension of [H, W, 3]
        """
        if self.model is None or img is None:
            return None
        # Preprocessing
        x = torch.from_numpy(img).permute(2, 0, 1).div(255).mul(2).add(-1)
        x_g = Grayscale()(x)

        x = x.unsqueeze(0).cuda()
        x_g = x_g.unsqueeze(0).cuda()
        hint = torch.zeros(1, 3, 16, 16).cuda()
        mask = torch.zeros(1, 1, 16, 16).cuda()

        # Inference
        code, code_log = self.get_visible_cods(x)

        with torch.no_grad():
            xrec, _ = self.model(x, x_g, hint, mask)

        # Postprocessing
        xrec = xrec.cpu().add(1).div(2).clamp(0, 1)[0]
        output = ToPILImage()(xrec)
        return code, output, code_log


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def load_model1024_imgnet():
    config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml",
                             display=False)
    model = load_vqgan(
        config1024,
        ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").cuda()
    return model

def load_model16384_imgnet():
    config16384 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml",
                              display=False)
    model16384 = load_vqgan(
        config16384,
        ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").cuda()
    return model

if __name__ == "__main__":

    model = load_model1024_imgnet()

    # DEFINE DATASET
    dataset = ImageFolder("~/dataset-local/imagenet/ILSVRC/Data/CLS-LOC/train",
                          transform=Compose([
                              ToTensor(),
                              Resize(256),
                              CenterCrop(256),
                          ]))
    dataloader = DataLoader(dataset, batch_size=32)

    # DEFINE DATASET
    with torch.no_grad():
        for i, (x, cls) in enumerate(tqdm(dataloader)):
            x = x.cuda()
            x = SignalProcessor.renorm_zero_1_to_m1_1(x)
            z, _, [_, _, indices] = model.encode(x)

            # xrec = model.decode(z)
            # xrec = SignalProcessor.renorm_m1_1_to_zero_1(xrec).clamp(0, 1)
            # ToPILImage()(xrec[1]).show()

            code: np.ndarray = indices.cpu().numpy()
            np.save("experiments/code-hist_imgnet1024/npys/code_%06d" % i, code)
