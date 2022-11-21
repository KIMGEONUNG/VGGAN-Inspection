import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config
from taming.modules.util import SOSProvider


def disabled_train(self, mode=True):
  """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
  return self


class HybridTransformer(pl.LightningModule):

  def __init__(
      self,
      transformer_config,
      first_stage_config,
      cond_stage_config,
      permuter_config=None,
      ckpt_path=None,
      ignore_keys=[],
      first_stage_key="image",
      cond_stage_key="depth",
      downsample_cond_size=-1,
      pkeep=1.0,
      sos_token=0,
      unconditional=False,
  ):
    super().__init__()
    self.be_unconditional = unconditional
    self.sos_token = sos_token
    self.first_stage_key = first_stage_key
    self.cond_stage_key = cond_stage_key
    self.init_first_stage_from_ckpt(first_stage_config)
    # self.init_cond_stage_from_ckpt(cond_stage_config)
    if permuter_config is None:
      permuter_config = {
          "target": "taming.modules.transformer.permuter.Identity"
      }
    self.permuter = instantiate_from_config(config=permuter_config)
    self.transformer = instantiate_from_config(config=transformer_config)

    dim_z_encoder_gray = first_stage_config["params"]["encoder_gray_config"][
        "z_channels"]

    self.linear4luma_g = nn.Conv2d(dim_z_encoder_gray,
                                   dim_z_encoder_gray,
                                   kernel_size=1)
    self.linear4rgb = nn.Conv2d(3,
                                dim_z_encoder_gray,
                                kernel_size=1,
                                bias=False)

    if ckpt_path is not None:
      self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    # self.downsample_cond_size = downsample_cond_size
    self.pkeep = pkeep

  def init_from_ckpt(self, path, ignore_keys=list()):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    for k in sd.keys():
      for ik in ignore_keys:
        if k.startswith(ik):
          self.print("Deleting key {} from state_dict.".format(k))
          del sd[k]
    self.load_state_dict(sd, strict=False)
    print(f"Restored from {path}")

  def init_first_stage_from_ckpt(self, config):
    model = instantiate_from_config(config)
    model = model.eval()
    model.train = disabled_train
    self.first_stage_model = model

  def forward(self, x, x_g, mask, hint):
    # one step to produce the logits
    _, z_indices = self.encode_to_z(x)
    a_indices = z_indices

    feat_g = self.first_stage_model.encoder_gray(x_g)
    feat_g = self.linear4luma_g(feat_g)
    feat_g = feat_g.view(*feat_g.shape[:-2], -1)  # flatten
    feat_g = feat_g.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

    mask = mask.view(*mask.shape[:-2], -1)  # flatten
    mask = mask.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

    hint = self.linear4rgb(hint)
    hint = hint.view(*hint.shape[:-2], -1)  # flatten
    hint = hint.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

    # target includes all sequence elements (no need to handle first one
    # differently because we are conditioning)
    target = z_indices
    # make the prediction
    logits, _ = self.transformer(a_indices[:, :-1], mask, hint, feat_g)

    # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
    logits = logits[:, feat_g.shape[1] - 1:]
    # logits = logits[:, feat_g.shape[1]:]

    return logits, target

  def top_k_logits(self, logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out

  @torch.no_grad()
  def sample(self,
             x,
             feat_g,
             mask,
             hint,
             steps,
             temperature=1.0,
             sample=False,
             top_k=None,
             callback=lambda k: None):
    # x = torch.cat((c, x), dim=1)
    block_size = self.transformer.get_block_size()
    assert not self.transformer.training
    for k in range(steps):
      # Prevent to mask on already known point
      mask[:, :max(0, k -1), :] = 1

      # stub callback, i.e. there is on operation.
      callback(k)
      # make sure model can see conditioning
      assert x.size(1) <= block_size
      # crop context if needed
      x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

      # crop the conditons, mask, hint,
      logits, _ = self.transformer(x_cond, mask, hint, feat_g)
      # pluck the logits at the final step and scale by temperature
      logits = logits[:, -1, :] / temperature
      # optionally crop probabilities to only the top k options
      if top_k is not None:
        logits = self.top_k_logits(logits, top_k)
      # apply softmax to convert to probabilities
      probs = F.softmax(logits, dim=-1)
      # sample from the distribution or take the most likely
      if sample:
        ix = torch.multinomial(probs, num_samples=1)
      else:
        _, ix = torch.topk(probs, k=1, dim=-1)
      # append to the sequence and continue
      x = torch.cat((x, ix), dim=1)
    return x

  @torch.no_grad()
  def encode_to_z(self, x):
    quant_z, _, info = self.first_stage_model.encode(x)
    indices = info[2].view(quant_z.shape[0], -1)
    indices = self.permuter(indices)
    return quant_z, indices

  @torch.no_grad()
  def encode_to_c(self, c):
    return c

  @torch.no_grad()
  def decode_to_img(self, index, zshape, feat_g):
    index = self.permuter(index,
                          reverse=True)  # [batch, 256(code)] why reverse?
    bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
    quant_z = self.first_stage_model.quantize.get_codebook_entry(
        index.reshape(-1), shape=bhwc)  # [batch, 256(code), 16, 16]

    quant_cat = torch.cat([quant_z, feat_g], dim=-3)

    x = self.first_stage_model.decode(quant_cat)
    return x

  @torch.no_grad()
  def log_images(self,
                 batch,
                 temperature=None,
                 top_k=None,
                 callback=None,
                 lr_interface=False,
                 **kwargs):
    log = dict()

    N = 4
    if lr_interface:
      x, x_g, mask, hint = self.get_xc(batch,
                                       N,
                                       diffuse=False,
                                       upsample_factor=8)
    else:
      x, x_g, mask, hint = self.get_xc(batch, N)
    x = x.to(device=self.device)
    x_g = x_g.to(device=self.device)
    mask = mask.to(device=self.device)
    hint = hint.to(device=self.device)

    quant_z, z_indices = self.encode_to_z(x)
    _, z_indices = self.encode_to_z(x)

    feat_g_origin = self.first_stage_model.encoder_gray(x_g)
    feat_g = self.linear4luma_g(feat_g_origin)
    feat_g = feat_g.view(*feat_g.shape[:-2], -1)  # flatten
    feat_g = feat_g.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

    mask = mask.view(*mask.shape[:-2], -1)  # flatten
    mask = mask.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

    hint = self.linear4rgb(hint)
    hint = hint.view(*hint.shape[:-2], -1)  # flatten
    hint = hint.transpose(-1, -2)  # [B,C,HW] --> [B,HW,C]

    # sample
    # shape of (4, 0), Note the dim of the second is zero
    z_start_indices = z_indices[:, :0]
    index_sample = self.sample(
        z_start_indices,
        feat_g,
        mask,
        hint,
        steps=z_indices.shape[1],
        temperature=temperature if temperature is not None else 1.0,
        sample=True,
        top_k=top_k if top_k is not None else 100,
        callback=callback if callback is not None else lambda k: None)

    x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape,
                                        feat_g_origin)

    # reconstruction
    x_rec = self.decode_to_img(z_indices, quant_z.shape, feat_g_origin)

    log["inputs"] = x
    log["reconstructions"] = x_rec

    log["samples_nopix"] = x_sample_nopix
    return log

  def get_input(self, key, batch):
    x = batch[key]
    if len(x.shape) == 3:
      x = x[..., None]
    if len(x.shape) == 4:
      x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    if x.dtype == torch.double:
      x = x.float()
    return x

  def get_xc(self, batch, N=None):
    x = self.get_input(self.first_stage_key, batch)
    c = self.get_input(self.cond_stage_key, batch)
    m = self.get_input("mask", batch)
    h = self.get_input("hint_grid", batch)
    if N is not None:
      x = x[:N]
      c = c[:N]
      m = m[:N]
      h = h[:N]
    return x, c, m, h

  def shared_step(self, batch, batch_idx):
    x, c, m, h = self.get_xc(batch)
    logits, target = self(x, c, m, h)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target.reshape(-1))
    return loss

  def training_step(self, batch, batch_idx):
    loss = self.shared_step(batch, batch_idx)
    self.log("train/loss",
             loss,
             prog_bar=True,
             logger=True,
             on_step=True,
             on_epoch=True)
    return loss

  # The function is also invoked for "Validation sanity check"
  def validation_step(self, batch, batch_idx):
    loss = self.shared_step(batch, batch_idx)
    self.log("val/loss",
             loss,
             prog_bar=True,
             logger=True,
             on_step=True,
             on_epoch=True)
    return loss

  def configure_optimizers(self):
    """
    Following minGPT:
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.transformer.named_modules():
      for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

        if pn.endswith('bias'):
          # all biases will not be decayed
          no_decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
          # weights of whitelist modules will be weight decayed
          decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
          # weights of blacklist modules will NOT be weight decayed
          no_decay.add(fpn)

    # special case for mask embedding
    decay.add('tok_mask')

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(
        inter_params
    ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
        str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": 0.01
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups,
                                  lr=self.learning_rate,
                                  betas=(0.9, 0.95))
    return optimizer
