#!/usr/bin/env python

import skimage
import gradio as gr
from cv2 import cv2
import torch
from glob import glob
from os.path import join, exists
from omegaconf import OmegaConf
import numpy as np
from torchvision.transforms import ToPILImage, Grayscale

from taming.util import load_model_from_config
from taming.algorithms import GrayConversion, SignalProcessor
from gui_config import MAP_PATH, CHOICES


class ReColorGUI(object):
    """Recolorization GUI System"""

    def __init__(self, share=True):
        self.share = share
        self.height = 300
        self.model = None
        self.hint = None
        self.gray = None

        self.gray_converter4view = GrayConversion(
            preprocess=lambda x: x / 255.0,
            postprocess=lambda x: (np.tile(x,
                                           (1, 1, 3)) * 255).astype('uint8'))

        self.methods_gray = [
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
            with gr.Box():
                with gr.Box(), gr.Row():
                    view_gt = gr.Image(
                        label="GT",
                        interactive=False).style(height=self.height)
                    with gr.Column():
                        view_gray = gr.Image(
                            label="Gray",
                            interactive=False).style(height=self.height)
                        with gr.Row():
                            btn_gray = gr.Button("Convert")
                            upload_gray = gr.UploadButton("Upload",
                                                          file_types=["image"],
                                                          file_count="single")
                            drop_gray = gr.Dropdown(choices=self.methods_gray,
                                                    value=self.methods_gray[0])
                    view_stroke = gr.ImagePaint(label="Stroke").style(
                        height=self.height)
                with gr.Box(), gr.Row():
                    with gr.Column():
                        view_hint = gr.Image(
                            label="hint",
                            interactive=False).style(height=self.height)
                        upload_hint = gr.UploadButton("Upload",
                                                      file_types=["image"],
                                                      file_count="single")
                    view_overlay = gr.Image(
                        label="overlay",
                        interactive=False).style(height=self.height)
                gr.Examples(sorted(glob("inputs/birds256/*")), inputs=view_gt)

            with gr.Box():
                with gr.Row():
                    path_log = gr.Dropdown(choices=CHOICES)
                    btn = gr.Button("Colorize").style(height=self.height)

                with gr.Row():
                    console = gr.Textbox(placeholder="No message",
                                         interactive=False)

            with gr.Row():
                view_output = gr.Image(
                    label="Output",
                    interactive=False).style(height=self.height)

            # Define GUI Events
            upload_hint.upload(self._upload_hint,
                               inputs=[upload_hint, view_gt],
                               outputs=[view_hint, view_overlay])
            upload_gray.upload(self._upload_gray,
                               inputs=[upload_gray],
                               outputs=[view_gray])
            view_gt.change(self._togray, view_gt, view_stroke)
            view_stroke.change(self._mk_hint, [view_stroke, view_gt],
                               [view_hint, view_overlay])
            path_log.change(self.set_model, inputs=[path_log], outputs=console)
            btn.click(self.predict, inputs=[view_gt], outputs=[view_output])
            view_gt.change(self._cvt_gray,
                           inputs=[view_gt, drop_gray],
                           outputs=[view_gray])
            btn_gray.click(self._cvt_gray,
                           inputs=[view_gt, drop_gray],
                           outputs=[view_gray])

    def launch(self):
        self.demo.launch(share=self.share)

    def _cvt_gray(self, x, method):
        if x is None:
            return None
        img, _ = self.gray_converter4view.gen_method3str(x, method)

        gray: torch.Tensor = torch.from_numpy(img[..., :1] / 255.0 * 2.0 - 1)
        self.gray = gray.permute(2, 0, 1).float()

        return img

    def _upload_gray(self, file):
        img = skimage.io.imread(file.name)
        gray: torch.Tensor = torch.from_numpy(img[..., :1] / 255.0 * 2.0 - 1)
        self.gray = gray.permute(2, 0, 1).float()
        return img

    def _upload_hint(self, file, gt):
        hint = skimage.io.imread(file.name)
        self.hint = hint  # Enroll hint

        # Make overlay
        alpha = 0.5
        hint_up = cv2.resize(hint,
                             gt.shape[:-1],
                             interpolation=cv2.INTER_NEAREST)
        mask = np.all(hint_up == [0, 0, 0], axis=-1)[..., None]

        hint_up = hint_up.astype('float')
        gt = gt.astype('float')
        blend = gt * mask + (alpha * gt + (1 - alpha) * hint_up) * (1 - mask)
        blend = blend.astype('uint8')

        return hint, blend

    def _mk_hint(self, x, gt):
        if x is None or gt is None:
            return None
        mask = (x[..., 0] == x[..., 1]) & (x[..., 1] == x[..., 2]) & (
            x[..., 2] == x[..., 0]) == False
        mask = mask[..., None]
        x = mask * x

        # Pooling
        x_r = self._dominant_pool_2d(x[..., 0])
        x_g = self._dominant_pool_2d(x[..., 1])
        x_b = self._dominant_pool_2d(x[..., 2])
        x = np.stack([x_r, x_g, x_b], axis=-1)
        self.hint = x  # Enroll hint

        # Make overlay
        alpha = 0.5
        hint_up = cv2.resize(x, gt.shape[:-1], interpolation=cv2.INTER_NEAREST)
        mask = np.all(hint_up == [0, 0, 0], axis=-1)[..., None]

        hint_up = hint_up.astype('float')
        gt = gt.astype('float')
        blend = gt * mask + (alpha * gt + (1 - alpha) * hint_up) * (1 - mask)
        blend = blend.astype('uint8')

        return x, blend

    def _dominant_pool_2d(self, spatial: np.ndarray, winsize=16):
        """
        Return a 2-D array with a pooling operation.
        The pooling operation is to select the most dominant value for each window.
        This assumes that the input 'spatial' has discrete values like index or lablel.
        To circumvent an use of iterative loop, we use a trick with one-hot encoding
        and 'skimage.measure.block_reduce' function.
        Parameters
        ----------
        spatial : int ndarray of shape (width, hight)
          The spatial is represented by int label, not one-hot encoding
        winsize : int, optional
          Length of sweeping window
        Returns
        -------
        pool : ndarray of shape (N,M)
          The pooling results.
        """
        num_seg = spatial.max() + 1
        one_hot = np.eye(num_seg)[spatial]
        sum_pooling = skimage.measure.block_reduce(one_hot,
                                                   (winsize, winsize, 1),
                                                   func=np.sum)
        pool = np.argmax(sum_pooling, axis=-1)
        return pool

    def _togray(self, x):
        if x is None:
            return None
        g = x.astype('float') / 255
        g = 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]
        g = np.tile(g[..., None], (1, 1, 3))
        return g

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
        message = "New model was set : ", path_log
        print(message)
        print("path_ckpt:", path_ckpt)
        return message

    def predict(self, img: np.ndarray):
        """
        img : range from 0 to 255 with unit8 and dimension of [H, W, 3]
        """
        if self.model is None:
            return None
        # Preprocessing
        x = torch.from_numpy(img).permute(2, 0, 1).div(255).mul(2).add(-1)
        if self.gray is None:
            x_g = Grayscale()(x)
        else:
            x_g = self.gray

        x = x.unsqueeze(0).cuda()
        x_g = x_g.unsqueeze(0).cuda()

        if self.hint is None:
            hint = torch.zeros(1, 3, 16, 16).cuda()
            mask = torch.zeros(1, 1, 16, 16).cuda()
        else:
            assert self.hint.shape == (16, 16, 3)
            mask = np.all(self.hint == [0, 0, 0], axis=-1) == False
            mask = mask[None, None, ...]
            mask = torch.from_numpy(mask)

            hint = torch.from_numpy(self.hint).permute(
                2, 0, 1).div(255).mul(2).add(-1)
            hint = hint[None, ...]

            mask = mask.to(torch.float32).cuda()
            hint = hint.to(torch.float32).cuda()

        # Inference
        with torch.no_grad():
            xrec, _ = self.model(x, x_g, hint, mask)

        # Postprocessing
        xrec = xrec.cpu().add(1).div(2).clamp(0, 1)[0]
        output = ToPILImage()(xrec)

        return output


if __name__ == "__main__":
    gui = ReColorGUI()
    gui.launch()
