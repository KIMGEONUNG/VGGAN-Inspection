#!/usr/bin/env python

import gradio as gr
import numpy as np
from taming.algorithms import GrayConversion


class GraysGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=True):
        self.share = share
        self.height = 300
        self.model = GrayConversion(preprocess=lambda x: x / 255.0,
                                    postprocess=lambda x:
                                    (np.tile(x,
                                             (1, 1, 3)) * 255).astype('uint8'))

        self.methods = [
            "classic",
            "scale",
            "heavy",
        ]

        # Define GUI Layout
        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box():
                with gr.Row():
                    view_gt = gr.Image(label="GT").style(height=self.height)
            gr.Examples([
                "inputs/ILSVRC2012_val_00002071_resize_256.JPEG",
                "inputs/ILSVRC2012_val_00005567_resize_256_256.JPEG",
                "inputs/ILSVRC2012_val_00045880_ccrop_0375_resize_256.JPEG",
            ],
                        inputs=view_gt)

            with gr.Box():
                with gr.Row():
                    methods = gr.Dropdown(choices=self.methods,
                                          value=self.methods[0])
                    num_infer = gr.Slider(minimum=1, maximum=32, value=16)
                    btn = gr.Button("Convert Gray").style(height=self.height)
            with gr.Box():
                with gr.Row():
                    gallery = gr.Gallery().style(grid=4)

            # Define Events
            btn.click(self.callback_inference,
                      inputs=[view_gt, methods, num_infer],
                      outputs=gallery)

    def launch(self):
        self.demo.launch(share=self.share)

    def callback_inference(self, x: np.ndarray, method: str, num_iter: int):
        return [
            self.model.gen_method3str(x, method) for _ in range(int(num_iter))
        ]


if __name__ == "__main__":
    gui = GraysGUI()
    gui.launch()
