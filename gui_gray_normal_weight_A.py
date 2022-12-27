#!/usr/bin/env python

import skimage
from glob import glob
import gradio as gr
import numpy as np
from taming.algorithms import GrayConversion


class GraysGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=False):
        self.share = share
        self.height = 300
        self.limit = 4

        # Define GUI Layout
        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box(), gr.Row():
                with gr.Column():
                    view_gt = gr.Image(
                        label="GT",
                        interactive=False).style(height=self.height)
                    upload = gr.UploadButton("Upload",
                                             file_types=["image"],
                                             file_count="single")
                view_gray = gr.Image(
                    label="Gray", interactive=False).style(height=self.height)

            gr.Examples(sorted(glob("inputs/birds256/*")), inputs=view_gt)

            with gr.Box(), gr.Row():
                coef_r = gr.Slider(minimum=-self.limit,
                                   maximum=self.limit,
                                   value=0,
                                   label="Coef R")
                coef_g = gr.Slider(minimum=-self.limit,
                                   maximum=self.limit,
                                   value=0,
                                   label="Coef G")
                coef_b = gr.Slider(minimum=-self.limit,
                                   maximum=self.limit,
                                   value=0,
                                   label="Coef B")

            # Define Events
            upload.upload(self._upload, inputs=upload, outputs=view_gt)
            for comp in [coef_r, coef_g, coef_b]:
                comp.change(
                    self.callback_inference,
                    inputs=[view_gt, coef_r, coef_g, coef_b],
                    outputs=[view_gray])

    def _upload(self, file):
        img = skimage.io.imread(file.name)
        return img

    def launch(self):
        self.demo.launch(share=self.share)

    def togray(self, x, r, g, b):
        x = x / 255.0
        w = np.array([r, g, b])
        x = x * 2 - 1
        x = (x @ w)[..., None]
        x = np.tanh(x)
        x = (x + 1) * 0.5
        x = np.tile(x, (1, 1, 3)) * 255
        x = x.astype('uint8')
        return x

    def callback_inference(self, x, r, g, b):
        if x is None:
            return None
        x = self.togray(x, r, g, b)
        return x


if __name__ == "__main__":
    gui = GraysGUI()
    gui.launch()
