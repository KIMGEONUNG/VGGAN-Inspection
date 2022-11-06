import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange
import yaml


class GumbelQuantize(nn.Module):

  def __init__(self,
               num_hiddens,
               embedding_dim,
               n_embed,
               straight_through=True,
               kl_weight=5e-4,
               temp_init=1.0,
               use_vqinterface=True,
               remap=None,
               unknown_index="random"):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.n_embed = n_embed

    self.straight_through = straight_through
    self.temperature = temp_init
    self.kl_weight = kl_weight

    self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
    self.embed = nn.Embedding(n_embed, embedding_dim)

    self.use_vqinterface = use_vqinterface

    self.remap = remap
    if self.remap is not None:
      self.register_buffer("used", torch.tensor(np.load(self.remap)))
      self.re_embed = self.used.shape[0]
      self.unknown_index = unknown_index  # "random" or "extra" or integer
      if self.unknown_index == "extra":
        self.unknown_index = self.re_embed
        self.re_embed = self.re_embed + 1
      print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
            f"Using {self.unknown_index} for unknown indices.")
    else:
      self.re_embed = n_embed

  def remap_to_used(self, inds):
    ishape = inds.shape
    assert len(ishape) > 1
    inds = inds.reshape(ishape[0], -1)
    used = self.used.to(inds)
    match = (inds[:, :, None] == used[None, None, ...]).long()
    new = match.argmax(-1)
    unknown = match.sum(2) < 1
    if self.unknown_index == "random":
      new[unknown] = torch.randint(
          0, self.re_embed, size=new[unknown].shape).to(device=new.device)
    else:
      new[unknown] = self.unknown_index
    return new.reshape(ishape)

  def unmap_to_all(self, inds):
    ishape = inds.shape
    assert len(ishape) > 1
    inds = inds.reshape(ishape[0], -1)
    used = self.used.to(inds)
    if self.re_embed > self.used.shape[0]:  # extra token
      inds[inds >= self.used.shape[0]] = 0  # simply set to zero
    back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
    return back.reshape(ishape)

  def forward(self, z, temp=None, return_logits=False):
    # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
    hard = self.straight_through if self.training else True
    temp = self.temperature if temp is None else temp

    logits = self.proj(z)
    if self.remap is not None:
      # continue only with used logits
      full_zeros = torch.zeros_like(logits)
      logits = logits[:, self.used, ...]

    soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
    if self.remap is not None:
      # go back to all entries but unused set to zero
      full_zeros[:, self.used, ...] = soft_one_hot
      soft_one_hot = full_zeros
    z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

    # + kl divergence to the prior loss
    qy = F.softmax(logits, dim=1)
    diff = self.kl_weight * torch.sum(
        qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

    ind = soft_one_hot.argmax(dim=1)
    if self.remap is not None:
      ind = self.remap_to_used(ind)
    if self.use_vqinterface:
      if return_logits:
        return z_q, diff, (None, None, ind), logits
      return z_q, diff, (None, None, ind)
    return z_q, diff, ind

  def get_codebook_entry(self, indices, shape):
    b, h, w, c = shape
    assert b * h * w == indices.shape[0]
    indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
    if self.remap is not None:
      indices = self.unmap_to_all(indices)
    one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1,
                                                                   2).float()
    z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
    return z_q


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


def Normalize(in_channels):
  return torch.nn.GroupNorm(num_groups=32,
                            num_channels=in_channels,
                            eps=1e-6,
                            affine=True)


def nonlinearity(x):
  # swish
  return x * torch.sigmoid(x)


class Upsample(nn.Module):

  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = torch.nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

  def forward(self, x):
    x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    if self.with_conv:
      x = self.conv(x)
    return x


class Downsample(nn.Module):

  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      # no asymmetric padding in torch conv, must do it ourselves
      self.conv = torch.nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)

  def forward(self, x):
    if self.with_conv:
      pad = (0, 1, 0, 1)
      x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
      x = self.conv(x)
    else:
      x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    return x


class ResnetBlock(nn.Module):

  def __init__(self,
               *,
               in_channels,
               out_channels=None,
               conv_shortcut=False,
               dropout,
               temb_channels=512):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels
    self.use_conv_shortcut = conv_shortcut

    self.norm1 = Normalize(in_channels)
    self.conv1 = torch.nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
    if temb_channels > 0:
      self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
    self.norm2 = Normalize(out_channels)
    self.dropout = torch.nn.Dropout(dropout)
    self.conv2 = torch.nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)
      else:
        self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

  def forward(self, x, temb):
    h = x
    h = self.norm1(h)
    h = nonlinearity(h)
    h = self.conv1(h)

    if temb is not None:
      h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

    h = self.norm2(h)
    h = nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        x = self.conv_shortcut(x)
      else:
        x = self.nin_shortcut(x)

    return x + h


class AttnBlock(nn.Module):

  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.norm = Normalize(in_channels)
    self.q = torch.nn.Conv2d(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
    self.k = torch.nn.Conv2d(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
    self.v = torch.nn.Conv2d(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
    self.proj_out = torch.nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

  def forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h * w)
    w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
    h_ = torch.bmm(
        v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return x + h_


class Encoder(nn.Module):

  def __init__(self,
               *,
               ch,
               ch_mult=(1, 2, 4, 8),
               num_res_blocks,
               attn_resolutions,
               dropout=0.0,
               resamp_with_conv=True,
               in_channels,
               resolution,
               z_channels,
               double_z=True,
               **ignore_kwargs):
    super().__init__()
    self.ch = ch
    self.temb_ch = 0
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels

    # downsampling
    self.conv_in = torch.nn.Conv2d(in_channels,
                                   self.ch,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    curr_res = resolution
    in_ch_mult = (1, ) + tuple(ch_mult)
    self.down = nn.ModuleList()
    for i_level in range(self.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = ch * in_ch_mult[i_level]
      block_out = ch * ch_mult[i_level]
      for i_block in range(self.num_res_blocks):
        block.append(
            ResnetBlock(in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout))
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      down = nn.Module()
      down.block = block
      down.attn = attn
      if i_level != self.num_resolutions - 1:
        down.downsample = Downsample(block_in, resamp_with_conv)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout)

    # end
    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(block_in,
                                    2 * z_channels if double_z else z_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

  def forward(self, x):
    # timestep embedding
    temb = None

    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1], temb)
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(self.down[i_level].downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # end
    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h


class Decoder(nn.Module):

  def __init__(self,
               *,
               ch,
               out_ch,
               ch_mult=(1, 2, 4, 8),
               num_res_blocks,
               attn_resolutions,
               dropout=0.0,
               resamp_with_conv=True,
               in_channels,
               resolution,
               z_channels,
               give_pre_end=False,
               **ignorekwargs):
    super().__init__()
    self.ch = ch
    self.temb_ch = 0
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels
    self.give_pre_end = give_pre_end

    # compute in_ch_mult, block_in and curr_res at lowest res
    in_ch_mult = (1, ) + tuple(ch_mult)
    block_in = ch * ch_mult[self.num_resolutions - 1]
    curr_res = resolution // 2**(self.num_resolutions - 1)
    self.z_shape = (1, z_channels, curr_res, curr_res)
    print("Working with z of shape {} = {} dimensions.".format(
        self.z_shape, np.prod(self.z_shape)))

    # z to block_in
    self.conv_in = torch.nn.Conv2d(z_channels,
                                   block_in,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout)

    # upsampling
    self.up = nn.ModuleList()
    for i_level in reversed(range(self.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = ch * ch_mult[i_level]
      for i_block in range(self.num_res_blocks + 1):
        block.append(
            ResnetBlock(in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout))
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in, resamp_with_conv)
        curr_res = curr_res * 2
      self.up.insert(0, up)  # prepend to get consistent order

    # end
    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(block_in,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

  def forward(self, z):
    # assert z.shape[1:] == self.z_shape[1:]
    self.last_z_shape = z.shape

    # timestep embedding
    temb = None

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = self.up[i_level].block[i_block](h, temb)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    # end
    if self.give_pre_end:
      return h

    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h


if __name__ == "__main__":

  # LOAD CONFIG
  config = load_config("configs/chroma_vqgan.yaml", display=False)
  config_params = config['model']['params']
  config_encoder = config_params['encoder']
  config_encoder_gray = config_params['encoder_gray']
  config_decoder = config_params['decoder']

  # INPUT SETUP
  x_gt = torch.randn(1, 3, 256, 256)
  x_gray = torch.randn(1, 1, 256, 256)

  # INIT MODELS
  encoder = Encoder(**config_encoder)
  encoder_gray = Encoder(**config_encoder_gray)
  decoder = Decoder(**config_decoder)

  z_channels = 512
  embed_dim = 512
  n_embed = 10 
  kl_weight = 1e-8
  temp_init = 1.0
  remap = None
  quantize = GumbelQuantize(z_channels, # ch_dim of z_hat
                            embed_dim,  # ch_dim of z, that is code
                            n_embed=n_embed, # The number of codes
                            kl_weight=kl_weight,
                            temp_init=temp_init,
                            remap=remap)


  # FEEDFORWARDS
  embd_chroma = encoder(x_gt)
  embd_chroma_vq, diff, idx = quantize(embd_chroma)
  embd_gray = encoder_gray(x_gray)
  embd = torch.cat([embd_chroma_vq, embd_gray], dim=-3)
  y = decoder(embd)

  print('Program finished')
