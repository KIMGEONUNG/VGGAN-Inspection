#!/usr/bin/env python

from glob import glob
import gradio as gr
import numpy as np
from taming.algorithms import GrayConversion


class GraysGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=False):
        self.share = share
        self.height = 300
        self.model = GrayConversion(preprocess=lambda x: x / 255.0,
                                    postprocess=lambda x:
                                    (np.tile(x,
                                             (1, 1, 3)) * 255).astype('uint8'))

        self.methods = [
            "classic",
            "scale",
            "scale_with_invert",
            "normal_weight_A",
            "normal_weight_A_equalsign",
            "normal_weight_B",
            "uniform_weight_A",
        ]

        # Define GUI Layout
        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box(), gr.Row():
                view_gt = gr.Image(label="GT",
                                   interactive=False).style(height=self.height)
            gr.Examples(sorted(glob("inputs/birds256/*")), inputs=view_gt)

            with gr.Box(), gr.Row():
                methods = gr.Dropdown(choices=self.methods,
                                      value=self.methods[0])
                num_infer = gr.Slider(minimum=1, maximum=1024, value=16)
                btn = gr.Button("Convert Gray").style(height=self.height)
            with gr.Box(), gr.Column():
                gallery = gr.Gallery().style(grid=4)
                with gr.Row():
                    avg = gr.Image(label="Average", interactive=False)
                    log = gr.Textbox()

            # Define Events
            btn.click(self.callback_inference,
                      inputs=[view_gt, methods, num_infer],
                      outputs=[gallery, avg, log])

    def launch(self):
        self.demo.launch(share=self.share)

    def callback_inference(self, x: np.ndarray, method: str, num_iter: int):
        if x is None:
            return None
        pairs = [
            self.model.gen_method3str(x, method) for _ in range(int(num_iter))
        ]
        imgs = [img for img, named_params in pairs]

        # CALCULATE AVERAGE
        avg = np.stack(imgs, axis=-1).astype('float').mean(axis=-1).astype('uint8')
        
        # EXTRACT LOG
        params = [str(named_params) for img, named_params in pairs]
        log = ""
        for i, p in enumerate(params):
            log += "%2d: %s\n" % (i, p)

        return imgs, avg, log


if __name__ == "__main__":
    gui = GraysGUI()
    gui.launch()
