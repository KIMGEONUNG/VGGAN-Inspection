import torch
import torch.nn.functional as F
from torchvision.transforms import Grayscale
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer


class ChromaVQ(pl.LightningModule):

    def __init__(
            self,
            encoder_config,
            encoder_gray_config,
            decoder_config,
            lossconfig,
            n_embed,
            embed_dim,
            ckpt_path=None,
            ignore_keys=[],
            image_key="image",
            gray_key="gray",
            colorize_nlabels=None,
            remap=None,
            sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.image_key = image_key
        self.gray_key = gray_key

        # Models
        self.encoder = Encoder(**encoder_config)
        self.encoder_gray = Encoder(**encoder_gray_config)
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
        self.image_key = image_key

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

    def encode(self, x_rgb):
        h = self.encoder(x_rgb)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x, x_g):
        quant, diff, _ = self.encode(x)
        feat_g = self.encoder_gray(x_g)

        quant_cat = torch.cat([quant, feat_g], dim=-3)

        dec = self.decode(quant_cat)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def synthesize_hint(self, x):
        return None

    def training_step(self, batch, batch_idx, optimizer_idx):
        # If you use multiple optimizers, training_step() will have an additional optimizer_idx parameter.
        # ref: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
        x = self.get_input(batch, self.image_key)
        x_g = self.get_input(batch, self.gray_key)

        ################## Synthesize Hint ####################################
        explicit_color = self.synthesize_hint(x)

        ################################################################### END

        xrec, qloss = self(x, x_g)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss,
                                            x,
                                            xrec,
                                            optimizer_idx,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")

            self.log("train/aeloss",
                     aeloss,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=True)
            self.log_dict(log_dict_ae,
                          prog_bar=False,
                          logger=True,
                          on_step=True,
                          on_epoch=True)
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
            self.log("train/discloss",
                     discloss,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=True)
            self.log_dict(log_dict_disc,
                          prog_bar=False,
                          logger=True,
                          on_step=True,
                          on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        x_g = self.get_input(batch, self.gray_key)

        xrec, qloss = self(x, x_g)
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
        self.log("val/rec_loss",
                 rec_loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        self.log("val/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.encoder_gray.parameters()) +
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

    def log_images(self, batch, **kwargs):  #  callback
        log = dict()
        x = self.get_input(batch, self.image_key)
        x_g = self.get_input(batch, self.gray_key)

        x, x_g = x.to(self.device), x_g.to(self.device)

        xrec, _ = self(x, x_g)

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
