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
from gui_config import MAP_PATH, CHOICES


class CodebookdGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=False):
        self.share = share
        self.height = 300
        self.model = None
        self.hint = None
        self.cmap = None

        # Define GUI Layout
        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box(), gr.Row():
                view_gt = gr.Image(label="GT",
                                   interactive=False).style(height=self.height)
                with gr.Column():
                    view_codebook = gr.Image(
                        label="Code",
                        interactive=False).style(height=self.height)
                    btn_getcode = gr.Button("Get Code")
                with gr.Column():
                    view_recon = gr.Image(
                        label="Recon.",
                        interactive=False).style(height=self.height)
                    btn_recon = gr.Button("Reconstruct from Code")
            gr.Examples(sorted(glob("inputs/birds256/*")), inputs=view_gt)
            with gr.Box(), gr.Column():
                with gr.Row():
                    choice_g = gr.Dropdown(
                        label="Gray",
                        choices=["Original", "Zeros", "Ones", "-Ones", "Gaussian"],
                        value="Original")
                    choice_gfeat = gr.Dropdown(
                        label="Gray Feature",
                        choices=["Original", "Zeros", "Ones", "Gaussian"],
                        value="Original")
                with gr.Row():
                    choice_model = gr.Dropdown(label="Model",
                                               choices=CHOICES)
                    log_ckpt = gr.Textbox(placeholder="No message",
                                          label="log ckpt",
                                          interactive=False)

            with gr.Box(), gr.Column():
                vqcodes = gr.Dataframe(
                    headers=['_'] * 16,
                    interactive=True,
                    row_count=(16, "fixed"),
                    col_count=(16, "fixed"),
                )
                with gr.Row():
                    btn_setall = gr.Button("Set all")
                    target_id = gr.Number()
            # Define GUI Events
            vqcodes.change(self.change_code,
                           inputs=vqcodes,
                           outputs=view_codebook)
            btn_setall.click(lambda x: [[x] * 16] * 16,
                             inputs=target_id,
                             outputs=vqcodes)
            choice_model.change(self.set_model,
                                inputs=[choice_model],
                                outputs=log_ckpt)
            btn_getcode.click(self.estimate_code,
                              inputs=view_gt,
                              outputs=[view_codebook, vqcodes])
            btn_recon.click(self.recon3code,
                            inputs=[view_gt, vqcodes, choice_g, choice_gfeat],
                            outputs=view_recon)

    def change_code(self, code):
        if code is None or self.cmap is None:
            return None
        code = code.to_numpy()
        code = self.cmap(code)[..., :-1]
        code = code.reshape(16, 16, -1)
        code = (code * 255).astype('uint8')
        return code

    def estimate_code(self, img):
        if self.model is None or img is None:
            return None
        # Preprocessing
        x = torch.from_numpy(img).permute(2, 0, 1).div(255).mul(2).add(-1)
        x = x.unsqueeze(0).cuda()

        # Estimatation
        h = self.model.encoder(x)
        h = self.model.quant_conv(h)
        quant, emb_loss, info = self.model.quantize(h)

        # Visualization
        code = info[2].cpu().numpy()
        code_log = code.reshape(16, 16)
        code = self.cmap(code)[..., :-1]
        code = code.reshape(16, 16, -1)
        code = (code * 255).astype('uint8')

        return code, code_log

    def recon3code(self, img, code, choice_g, choice_gfeat):
        if self.model is None or img is None:
            return None
        # Preprocessing
        x = torch.from_numpy(img).permute(2, 0, 1).div(255).mul(2).add(-1)
        x_g = Grayscale()(x)

        x = x.unsqueeze(0).cuda()
        x_g = x_g.unsqueeze(0).cuda()

        if choice_g == "Original":
            pass
        elif choice_g == "Zeros":
            x_g = torch.zeros_like(x_g)
        elif choice_g == "Ones":
            x_g = torch.ones_like(x_g)
        elif choice_g == "-Ones":
            x_g = - torch.ones_like(x_g)
        elif choice_g == "Gaussian":
            x_g = torch.randn_like(x_g)

        code = torch.from_numpy(code.to_numpy()).cuda()

        # Estimation
        feat_g = self.model.encoder_gray(x_g)

        if choice_gfeat == "Original":
            pass
        elif choice_gfeat == "Zeros":
            feat_g = torch.zeros_like(feat_g)
        elif choice_gfeat == "Ones":
            feat_g = torch.ones_like(feat_g)
        elif choice_gfeat == "Gaussian":
            feat_g = torch.randn_like(feat_g)

        quant = self.model.quantize.get_codebook_entry(code, (1, 16, 16, 256))
        quant = torch.cat([quant, feat_g], dim=-3)
        xrec = self.model.decode(quant)

        # Postprocessing
        xrec = xrec.cpu().add(1).div(2).clamp(0, 1)[0]
        output = ToPILImage()(xrec)
        return output

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
    def get_visible_codes(self, x):
        h = self.model.encoder(x)
        h = self.model.quant_conv(h)  # After that we have to use mask and hint
        quant, emb_loss, info = self.model.quantize(h)
        code = info[2].cpu().numpy()
        code_log = code.reshape(16, 16)
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
        code, code_log = self.get_visible_codes(x)

        with torch.no_grad():
            xrec, _ = self.model(x, x_g, hint, mask)

        # Postprocessing
        xrec = xrec.cpu().add(1).div(2).clamp(0, 1)[0]
        output = ToPILImage()(xrec)
        return code, output, code_log


if __name__ == "__main__":
    gui = CodebookdGUI()
    gui.launch()
