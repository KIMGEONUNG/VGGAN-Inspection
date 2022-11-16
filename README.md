# Chroma-VQGAN

Vector Quantized Generative Adversarial Network (VGGAN) achieves a performant fusion architecture of CNN and Transformer, now become an prevalent backbone network.
In this work, Let's inspect the details of VQGAN and get some insights to reproduce Unicolor.

## Fast links

- [Reproduce Memo](#reproduce-memo)
- [Reconstruction](#reconstruction)
- [Training VQGAN](#training-vqgan)
- [Issues](#issues)

## Reproduce Memo
<a id="reproduce-memo"></a>

- BugFix
  - <span style="color:red">Move the codes of arbatrary grayscale conversion to the dataloader</span>
- Refactoring
  - Create a file only for Chroma-VQ class <span style="color:gray">(Undo)</span>
- Experiments on Chroma-VQ 1st stage 
  - Multiple inferences on the arbitrary grayscales which are converted from a single color image. <span style="color:gray">(Undo)</span>
- Reproduce 2nd stage training codes for Chroma-VQGAN 
  - Implement training codes 
    - Implement Data loading <span style="color:red">(WIP)</span>
    - Implement Hybrid-Transformer <span style="color:red">(WIP)</span>
      - As the training example includes conditional input, it seems better to preserve original structure and to implant conditional code.
      - Understand BERT-style scheme
      - Add MASK TOKEN
      - Understand spatial concatenate of $f_g$ and $f_h$
      - Understand Hint-point scheduling
    - Implement hint sampler <span style="color:red">(WIP)</span>
- Add arbitary intensity generation to 1st stage <span style="color:green">(Done)</span>
- Connect UI to model <span style="color:gray">(Undo)</span>

<!-- <figure> -->
<!-- <img src="assets/teaser.png" alt="fail" style="width:100%"> -->
<!-- <figcaption align = "center"><b>Fig.1 VGGAN</b></figcaption> -->
<!-- </figure> -->

<figure>
<img src="assets/chroma-vqgan.png" alt="fail" style="width:100%">
<figcaption align = "center"><b>Fig.2 UniColor framework overview</b></figcaption>
</figure>

## Fixed-luminance Assumption in Colorization Problem

Colorization problem has been defined as a mapping from a luminance map with optional guidances to two-channel chrominance, i.e., the luminance map is always fixed at the final results.
However, this fixed-luminance assumption is far from the practical colorization scenario because there are many use cases which require to modulate luminance map, such as conversion from a black suit to a white one.
Unfortunately, the luminance modification is not trivial due to the challenges as follows.
- To make low intensity brighter can produces some artifacts because the low intensity regions usually has degraded fine-details.
some additional restoration techniques can be required for color editing
- The exact region which should be change with respect to luminance is ambiguous
- <span style="color:red">One more find!</span>


## Stochastic Intensity Conversion  

Consider a natural image $I_{RGB} = [R, G, B] \in [0,1]^3$.
To evade non-negativity, the rearraged image as
$$
 I^{*}_{RGB}=2I_{RGB}-1
$$
where $I^{*}_{RGB} = [R^*,G^*,B^*] \in [-1,1]^3$.
To produce arbitrary intensities with single channel, the forumlation is as 

$$
 I^{*}_{g} = x_1 R^* + x_2 G^* + x_3 B^*,\quad \text{where}~x \sim \mathcal{N(0, 1)}.
$$

Before restored to original image domain of $[0,1]^3$, The hyperbolic tangent is used to fix the low contrast of $I^{*}_{g}$.
Fianlly, the formulation is as

$$
 I_{g} = \frac{1 + tanh(I^{*}_{g})}{2}
$$

## Stage2 Structure

FFHQ pretrained model used _Net2NetTransformer_ class.

```python
# taming/models/cond_transformer.py
class Net2NetTransformer(pl.LightningModule):
  ...
```

## Autoregressive Sampling

The following code show the process of autoregressive sampling in quantized feature space.
Note that the next quantized code is sampled by the estimated mutinomial distribution.

```python
# taming/modules/transformer/mingpt.py
@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None):
    # x is conditioning
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1))
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1) # Note that sampling from multinomial
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample
```

## Idea Memo
- random grascale transfer? usint 1x1 convolution

## Pytorch Lightning Call Sequence

- configure_optimizers
- validation_step
- training_step

## Gradient Copy

Gradient copy overcoming non-differientiable operation of the _argmin_ can be implemented like below.

```python
# use value of z_q and gradient of z
z_q = z + (z_q - z).detach()
```

## Reconstruction
<a id="reconstruction"></a>

The author provides an [example code](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb#scrollTo=3RxdhDGtyJ4q) to enable checking the reconstruction quality. 
I move and refine the example code into a file, recon.py. 
The example uses three types of variants `VQGAN(f8,8192)`, `VGGAN(f16, 16384)`, `VQGAN(f16, 1024)`.
The notation `f{N}` means the spatial resolution of feature as K/N by K/N w.r.t. the input image size of K by K, that is `f8` and `f16` means 32 by 32 and 16 by 16 spatial resolution from 256 by 256 imput image.
The second number refers to the number of codebook entries.
Although not described, the dimensionality of codebook entries are generally 256.
A result example is as
![sampe](assets/cmp1.jpg) 
As shown the figure, the high spatial dimension, VQGAN(f8, 8192), shows the most high fidelity w.r.t. image structure.

Although VQGAN shows the performant reconstruction capability, they fail to restore fine details like face elements as shown in below figure.
![sampe](assets/cmp2.jpg) 

### Input Image Preprocessing

They use input images reranged from [0, 1] to [-1, 1]. 
The code example is as below.

```python
def preprocess_vqgan(x):
  x = 2. * x - 1.
  return x
```

### Output Image Postprocessing

They assumes that the output image has a range [-1, 1], but not strict, so requires cliping process.
The code example is as below.

```python
x = torch.clamp(x, -1., 1.)
x = (x + 1.) / 2.
```


### Objectives

As a part of objectives, they used same loss terms as used in VQVAE as  
$\mathcal{L}_{VQ}=\mathcal{L}_{recon} + \mathcal{L}_{code} + \mathcal{L}_{commit}$  
$\mathcal{L}_{recon}=||x - \hat{x}||^2$ for supervison,  
$\mathcal{L}_{code}=||sg[E(x)] - z_q]|^{2}_{2}$ to optimize predefined codebooks  
$\mathcal{L}_{commit}=||sg[z_q] - E(x)||^{2}_{2}$  


### Model Components

The VQModel consists of Encoder and Decoder and the details as following
<details>
<summary>Details</summary>

```bash
GumbelVQ(
  (encoder): Encoder(
    (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (down): ModuleList(
      (0): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (1): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (2): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (3): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList(
          (0): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
    )
    (mid): Module(
      (block_1): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (attn_1): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (block_2): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
    (conv_out): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (decoder): Decoder(
    (conv_in): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (mid): Module(
      (block_1): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (attn_1): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (block_2): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (up): ModuleList(
      (0): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
      )
      (1): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (2): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (3): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList(
          (0): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (2): AttnBlock(
            (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
            (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (upsample): Upsample(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
    (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (loss): DummyLoss()
  (quantize): GumbelQuantize(
    (proj): Conv2d(256, 8192, kernel_size=(1, 1), stride=(1, 1))
    (embed): Embedding(8192, 256)
  )
  (quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (post_quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
)
```
</details>

<details>

The below is the model specification of the transformer which autoregressively samples the human faces.

<summary>Details</summary>

```bash
Net2NetTransformer(
  (first_stage_model): VQModel(
    (encoder): Encoder(
      (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (down): ModuleList(
        (0): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (1): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (2): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (3): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (downsample): Downsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
          )
        )
        (4): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList(
            (0): AttnBlock(
              (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
              (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): AttnBlock(
              (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
              (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            )
          )
        )
      )
      (mid): Module(
        (block_1): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (attn_1): AttnBlock(
          (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (block_2): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
      (conv_out): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (decoder): Decoder(
      (conv_in): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (mid): Module(
        (block_1): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (attn_1): AttnBlock(
          (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (block_2): ResnetBlock(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (up): ModuleList(
        (0): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
        )
        (1): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (2): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (3): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (4): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList(
            (0): AttnBlock(
              (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
              (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): AttnBlock(
              (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
              (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            )
            (2): AttnBlock(
              (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
              (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
              (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            )
          )
          (upsample): Upsample(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (loss): DummyLoss()
    (quantize): VectorQuantizer2(
      (embedding): Embedding(1024, 256)
    )
    (quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (post_quant_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (cond_stage_model): SOSProvider()
  (permuter): Identity()
  (transformer): GPT(
    (tok_emb): Embedding(1024, 1664)
    (drop): Dropout(p=0.0, inplace=False)
    (blocks): Sequential(
      (0): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (1): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (2): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (3): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (4): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (5): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (6): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (7): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (8): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (9): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (10): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (11): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (12): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (13): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (14): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (15): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (16): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (17): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (18): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (19): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (20): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (21): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (22): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
      (23): Block(
        (ln1): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (ln2): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (key): Linear(in_features=1664, out_features=1664, bias=True)
          (query): Linear(in_features=1664, out_features=1664, bias=True)
          (value): Linear(in_features=1664, out_features=1664, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (resid_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1664, out_features=1664, bias=True)
        )
        (mlp): Sequential(
          (0): Linear(in_features=1664, out_features=6656, bias=True)
          (1): GELU()
          (2): Linear(in_features=6656, out_features=1664, bias=True)
          (3): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1664,), eps=1e-05, elementwise_affine=True)
    (head): Linear(in_features=1664, out_features=1024, bias=False)
  )
)
```
</details>

## Sampling

The author provides some commands to sample images from pretrained GAN models.


```sh
# S-FLCKR
python scripts/sample_conditional.py -r logs/2020-11-09T13-31-51_sflckr/

# ImageNet
python scripts/sample_fast.py -r logs/2021-04-03T19-39-50_cin_transformer/ -n 10 -k 600 -t 1.0 -p 0.92 --batch_size 10

# FFHQ
python scripts/sample_fast.py -r logs/2021-04-23T18-19-01_ffhq_transformer/

# COCO
CUDA_VISIBLE_DEVICES=1 streamlit run scripts/sample_conditional.py -- -r logs/2021-01-20T16-04-20_coco_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.coco.Examples}}}"
```

To be honest, the sampling quality is quite dissapointing, which implies that the representation is performant, but the sampling process is comparably inferior.
The below shows some results samples from FFHQ pretrained model.

![smp1](assets/sample-ffhq1.png) 
![smp2](assets/sample-ffhq2.png) 
![smp3](assets/sample-ffhq3.png) 


## Training VQGAN
<a id="training-vqgan"></a>

We need both of 1st and 2nd training codes, becuase UniColor use them for training.

### Training for 1st Stage

1. put your .jpg files in a folder `your_folder`
2. create 2 text files a `xx_train.txt` and `xx_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
3. adapt `configs/custom_vqgan.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU. 


```sh
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,
```

### Training for 2nd Stage

```sh
python main.py --base configs/coco_scene_images_transformer.yaml -t True --gpus 0,
python main.py --base configs/open_images_scene_images_transformer.yaml -t True --gpus 0,
```


Download first-stage models [COCO-8k-VQGAN](https://heibox.uni-heidelberg.de/f/78dea9589974474c97c1/) for COCO or [COCO/Open-Images-8k-VQGAN](https://heibox.uni-heidelberg.de/f/461d9a9f4fcf48ab84f4/) for Open Images.
Change `ckpt_path` in `data/coco_scene_images_transformer.yaml` and `data/open_images_scene_images_transformer.yaml` to point to the downloaded first-stage models.
Download the full COCO/OI datasets and adapt `data_path` in the same files, unless working with the 100 files provided for training and validation suits your needs already.

Code can be run with
`python main.py --base configs/coco_scene_images_transformer.yaml -t True --gpus 0,`
or
`python main.py --base configs/open_images_scene_images_transformer.yaml -t True --gpus 0,`


## Output Sequence w.r.t. Epoches
 The first row is GT image and the next things are the reconstructuion results of every 10 epoches.
Referred the results, the epoch number 50 is enough to check the feasibility.
After 150 epoch, there are few changes.

![aa](assets/training-stage1-each-epoch/inputs_gs-406999_e-000999_b-000000.png) 
![aa](assets/training-stage1-each-epoch/reconstructions_gs-000406_e-000000_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-004476_e-000010_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-008546_e-000020_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-012616_e-000030_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-016686_e-000040_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-020756_e-000050_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-024826_e-000060_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-028896_e-000070_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-032966_e-000080_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-037036_e-000090_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-041106_e-000100_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-045176_e-000110_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-049246_e-000120_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-053316_e-000130_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-057386_e-000140_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-061456_e-000150_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-065526_e-000160_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-069596_e-000170_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-073666_e-000180_b-000000.png)
![aa](assets/training-stage1-each-epoch/reconstructions_gs-077736_e-000190_b-000000.png)


## Issues
<a id="issues"></a>

#### omegaconf.errors.ConfigAttributeError: Missing key logger <span style="color:green">(Solved)</span>

The fundamental reason is package version mismatch to _pytorch-lightening_ and _omegaconfig_.
[An issue article](https://github.com/CompVis/taming-transformers/issues/72#issuecomment-875757912) introduced some solutions.
I tried many solution, but eventually failed with subsequent issues.
Not elegant, but working solution is as below

```sh
pip install pytorch-lightning==1.0.8 omegaconf==2.0.0
```

The error message is as

<details>
<summary>messages</summary>

```
Global seed set to 23
Running on GPUs 0,
Working with z of shape (1, 256, 16, 16) = 65536 dimensions.
loaded pretrained LPIPS loss from taming/modules/autoencoder/lpips/vgg.pth
VQLPIPSWithDiscriminator running with hinge loss.
Traceback (most recent call last):
File "main.py", line 465, in <module>
logger_cfg = lightning_config.logger or OmegaConf.create()
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 355, in __getattr__
self._format_and_raise(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/base.py", line 231, in _format_and_raise
format_and_raise(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/_utils.py", line 900, in format_and_raise
_raise(ex, cause)
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/_utils.py", line 798, in _raise
raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 351, in __getattr__
return self._get_impl(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 442, in _get_impl
node = self._get_child(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/basecontainer.py", line 73, in _get_child
child = self._get_node(
File "/home/comar/anaconda3/lib/python3.8/site-packages/omegaconf/dictconfig.py", line 480, in _get_node
raise ConfigKeyError(f"Missing key {key!s}")
omegaconf.errors.ConfigAttributeError: Missing key logger
full_key: logger
object_type=dict
)))))
```

</details>

#### ModuleNotFoundError: No module named 'main' <span style="color:green">(Solved)</span>

Install repository itself.

```
install pip -e .
```
