import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Grayscale
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

from torchvision.transforms import ToPILImage, Resize
from torchvision.transforms import InterpolationMode
import wandb


class ChromaVQ(pl.LightningModule):

    def __init__(
            self,
            encoder_config,
            encoder_gray_config,
            decoder_config,
            lossconfig,
            n_embed,
            embed_dim,
            vqhint,
            cond_gray_feat=False,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            gray_key="gray",
            hint_key="hint",
            mask_key="mask",
            colorize_nlabels=None,
            remap=None,
            sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.image_key = image_key
        self.gray_key = gray_key
        self.hint_key = hint_key
        self.mask_key = mask_key
        self.vqhint = vqhint
        self.cond_gray_feat = cond_gray_feat

        # Models
        self.encoder = Encoder(**encoder_config)
        self.encoder_gray = Encoder(**encoder_gray_config)
        self.color2embd = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.quant_conv = torch.nn.Conv2d(encoder_config["z_channels"],
                                          embed_dim, 1)

        self.quantize = VectorQuantizer(n_embed,
                                        embed_dim,
                                        beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)

        self.post_quant_conv = torch.nn.Conv2d(decoder_config["z_channels"],
                                               decoder_config["z_channels"], 1)
        self.decoder = Decoder(**decoder_config)
        self.loss = instantiate_from_config(lossconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):  # For retrain
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x_rgb, feat_g, hint_embd, mask):
        h = self.encoder(x_rgb)
        h = self.quant_conv(h)  # After that we have to use mask and hint

        if self.vqhint:
            h = mask * hint_embd + (1 - mask) * h
            quant, emb_loss, info = self.quantize(h)
        else:
            quant, emb_loss, info = self.quantize(h)
            quant = mask * hint_embd + (1 - mask) * quant

        if self.cond_gray_feat:
            feat_g = feat_g + mask * hint_embd

        quant = torch.cat([quant, feat_g], dim=-3)

        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x, x_g, hint, mask):
        hint_embd = self.color2embd(hint)
        feat_g = self.encoder_gray(x_g)

        quant, diff, _ = self.encode(x, feat_g, hint_embd, mask)

        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        # If you use multiple optimizers, training_step() will have an additional optimizer_idx parameter.
        # ref: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
        x = self.get_input(batch, self.image_key)
        x_g = self.get_input(batch, self.gray_key)
        hint = self.get_input(batch, self.hint_key)
        mask = self.get_input(batch, self.mask_key)

        xrec, qloss = self(x, x_g, hint, mask)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss,
                                            x,
                                            xrec,
                                            optimizer_idx,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")
            self.log(
                "train/aeloss",
                aeloss,
                rank_zero_only=True,
            )
            self.log_dict(
                log_dict_ae,
                rank_zero_only=True,
            )
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train")

            self.log(
                "train/discloss",
                discloss,
                logger=True,
                rank_zero_only=True,
            )

            self.log_dict(
                log_dict_disc,
                logger=True,
                rank_zero_only=True,
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        x_g = self.get_input(batch, self.gray_key)
        hint = self.get_input(batch, self.hint_key)
        mask = self.get_input(batch, self.mask_key)

        xrec, qloss = self(x, x_g, hint, mask)
        aeloss, log_dict_ae = self.loss(qloss,
                                        x,
                                        xrec,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val")

        discloss, log_dict_disc = self.loss(qloss,
                                            x,
                                            xrec,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")
        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/rec_loss", rec_loss)
        self.log("val/aeloss", aeloss)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        if batch_idx % 5:
            self._log_images(x, x_g, hint, mask, xrec)

        return self.log_dict

    def _log_images(self, x, x_g, hint, mask, recon):

        x = x.add(1).div(2).clamp(0, 1)
        x_g = x_g.repeat(1, 3, 1, 1).add(1).div(2).clamp(0, 1)
        hint = hint.add(1).div(2).clamp(0, 1)
        recon = recon.add(1).div(2).clamp(0, 1)

        size = x.shape[2:4]
        hint = Resize(size,
                      interpolation=InterpolationMode.NEAREST)(hint * mask)

        imgs = torch.cat([x, x_g, hint, recon], dim=-2)

        self.logger.log_image(key="Results",
                              images=[ToPILImage()(img) for img in imgs],
                              step=self.global_step)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.encoder_gray.parameters()) +
                                  list(self.color2embd.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr,
                                  betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,
                                    betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
