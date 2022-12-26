#!/usr/bin/env python

import gradio as gr
import torch
from glob import glob
from os.path import join, exists
from omegaconf import OmegaConf
import numpy as np
from torchvision.transforms import ToPILImage, Grayscale

from taming.util import load_model_from_config
import matplotlib.pyplot as plt


class CodebookdGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=True):
        self.share = share
        self.height = 300
        self.model = None
        self.hint = None

        # Define GUI Layout
        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box():
                with gr.Row():
                    view_gt = gr.Image(
                        label="GT",
                        interactive=False).style(height=self.height)
                    view_codebook = gr.Image(
                        label="Code",
                        interactive=False).style(height=self.height)
                    view_recon = gr.Image(
                        label="Recon.",
                        interactive=False).style(height=self.height)
                with gr.Row():
                    upload_button = gr.UploadButton(
                        "Click to Upload a hint image",
                        file_types=["image"],
                        file_count="single")
                gr.Examples(sorted(glob("inputs/birds256/*")), inputs=view_gt)

            with gr.Box():
                with gr.Row():
                    path_log = gr.Dropdown(choices=[
                        "Baseline",
                        "RandGray",
                        "VQHint",
                        "VQHint+RandGray",
                        "ScaleGray",
                    ])
                    btn = gr.Button("Colorize").style(height=self.height)

                with gr.Column():
                    log_ckpt = gr.Textbox(placeholder="No message",
                                          label="log ckpt",
                                          interactive=False)
                    log_code = gr.Textbox(placeholder="No message",
                                          label="log code",
                                          interactive=False)

            # Define GUI Events
            path_log.change(self.set_model,
                            inputs=[path_log],
                            outputs=log_ckpt)
            btn.click(self.predict,
                      inputs=[view_gt],
                      outputs=[view_codebook, view_recon, log_code])

    def launch(self):
        self.demo.launch(share=self.share)

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

        message = "New model was set %s with n_embed %d" % (path_ckpt,
                                                            self.n_embed)
        print(message)
        print("path_ckpt:", path_ckpt)
        return message

    @torch.no_grad()
    def get_visible_cods(self, x):
        h = self.model.encoder(x)
        h = self.model.quant_conv(
            h)  # After that we have to use mask and hint
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


if __name__ == "__main__":
    gui = CodebookdGUI()
    gui.launch()
