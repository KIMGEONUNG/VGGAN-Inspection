#!/usr/bin/env python

import skimage
import gradio as gr
from cv2 import cv2
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
from torchvision.transforms import ToPILImage, Grayscale

from taming.util import (get_obj_from_str, instantiate_from_config,
                         load_model_from_config)
from taming.algorithms import HintSampler


class HintGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=False):
        self.share = share
        self.height = 300
        self.sampler = None
        self.hint = None

        self.sampler = HintSampler()

        methods_hint = ["Naive"]
        methods_mask = ["Uniform"]
        methods_blend = ["Naive"]

        # Define GUI Layout
        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box():
                with gr.Row():
                    view_gt = gr.Image(
                        label="GT",
                        interactive=False).style(height=self.height)
                with gr.Row():
                    upload_button = gr.UploadButton(
                        "Click to Upload a hint image",
                        file_types=["image"],
                        file_count="single")
                gr.Examples(sorted(glob("inputs/birds256/*")), inputs=view_gt)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            view_hint = gr.Image(
                                label="Hint",
                                interactive=False).style(height=self.height)
                            method_hint = gr.Dropdown(choices=methods_hint,
                                                      value=methods_hint[0])
                            btn_hint = gr.Button("Generate")
                        with gr.Column():
                            view_mask = gr.Image(
                                label="Mask",
                                interactive=False).style(height=self.height)
                            method_mask = gr.Dropdown(choices=methods_mask,
                                                      value=methods_mask[0])
                            btn_mask = gr.Button("Generate")
                        with gr.Column():
                            view_blend = gr.Image(
                                label="Blend",
                                interactive=False).style(height=self.height)
                            method_blend = gr.Dropdown(choices=methods_blend,
                                                       value=methods_blend[0])
                            btn_blend = gr.Button("Generate")
            # Define GUI Events
            btn_hint.click(self.gen_hint,
                           inputs=[view_gt, method_hint],
                           outputs=view_hint)
            btn_mask.click(self.gen_mask,
                           inputs=[view_gt, method_mask],
                           outputs=view_mask)
            btn_blend.click(self.gen_blend,
                            inputs=[view_hint, view_mask],
                            outputs=view_blend)
            upload_button.upload(lambda x: x.name,
                                 inputs=upload_button,
                                 outputs=view_gt)

            view_gt.change(self.gen_hint,
                           inputs=[view_gt, method_hint],
                           outputs=view_hint)
            view_hint.change(self.gen_blend,
                             inputs=[view_hint, view_mask],
                             outputs=view_blend)
            view_mask.change(self.gen_blend,
                             inputs=[view_hint, view_mask],
                             outputs=view_blend)

    def launch(self):
        self.demo.launch(share=self.share)

    def prep(self, x):
        return x / 255.0

    def post(self, x):
        return (x * 255.0).astype("uint8")

    def gen_mask(self, gt, name_method):
        mask = np.random.binomial(n=1, p=0.5, size=(16, 16))
        mask = np.tile(mask[..., None], (1, 1, 3)) * 255
        return mask

    def gen_blend(self, hint, mask):
        if hint is None or mask is None:
            return None
        return (hint * (mask / 255)).astype("uint8")

    def gen_hint(self, gt, name_method):
        if gt is None:
            return None
        x = self.prep(gt)
        hint = self.sampler(x)
        hint = self.post(hint)
        return hint


if __name__ == "__main__":
    gui = HintGUI()
    gui.launch()
