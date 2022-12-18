import numpy as np
import torch
from PIL import Image
import random
import skimage
from skimage.segmentation import slic


class HintSampler():

    def __init__(self):
        pass

    def gen_mask_hint_label(
        dim_spatial=(16, 16),
        prop_mask_all=0.03,
        num_mask_from=16,
        num_mask_to=256,
        prop_hint=0.7,
        num_hint_from=1,
        num_hint_to=16,
    ):
        """
        The function sample masked and hint regions.
        Note that the hint points are sampled among masked regions.

        Parameters
        ----------
        dim_spatial : an tuple of shape (W, H)
        prop_mask_all : float for probability
        num_mask_from : int number for minimum value of the number of sampling mask
        num_mask_to : int number for upper bound of the number of sampling mask
        prop_hint : float for probability
        num_hint_from : int number for minimum value of the number of sampling hint
        num_hint_to : int number for minimum value of the number of sampling hint

        Returns
        -------
        mask : ndarray of shape (W, H)
          the mask has three types of value
           0 --> masked points
           1 --> unmasked points, i.e. preserve original values
          -1 --> the points using color hint
        """

        dim_spatial_flat = dim_spatial[0] * dim_spatial[1]

        # MASK SYNTHESIS
        mask = np.ones(dim_spatial_flat).astype('int')
        is_all_mask = np.random.binomial(1, p=prop_mask_all, size=1)[0]
        if is_all_mask:
            idx_mask = random.sample(range(0, dim_spatial_flat),
                                     dim_spatial_flat)
        else:
            num_mask = np.random.randint(num_mask_from, num_mask_to + 1)
            idx_mask = random.sample(range(0, dim_spatial_flat), num_mask)

        mask[idx_mask] = 0

        # HINT SELECTION
        is_hint = np.random.binomial(1, p=prop_hint, size=1)[0]
        if is_hint:
            num_hint = np.random.randint(num_hint_from, num_hint_to + 1)
            idx_hint = random.sample(idx_mask, num_hint)
            mask[idx_hint] = -1

        mask = mask.reshape(*dim_spatial)

        return mask

    def dominant_pool_2d(self, spatial: np.ndarray, winsize=16):
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

    def gen_sp2rgbmean(self, image: np.ndarray, segments: np.ndarray):
        """
        Generate a mapping from superpixel index to the corresponding RGB mean value

        Parameters
        ----------
        image : image ndarray of shape (width, hight, dim_spectrum)
          The spatial is represented by int label, not one-hot encoding
        segments : int ndarray of shape (width, hight)
          Segmentation labels for each pixel

        Returns
        -------
        sp2rgb : ndarray of shape (N, dim_spectrum)
          Mapping of superpixel index to RGB mean value.
          N is the number of segmentation label.
        """

        # Define required values
        num_seg = segments.max() + 1
        dim_spatial = image.shape[:-1]  # (w, h)
        axis_spatial = tuple(range(len(dim_spatial)))
        dim_spectral = 3  # RGB

        # Convert label to one-hot encoding
        one_hot = np.eye(num_seg)[segments]
        one_hot_sum = one_hot.sum(axis_spatial)

        # Calculate mean RGB value according to the label
        pix_lab_rgb = one_hot.reshape(*dim_spatial, num_seg,
                                      1) * image.reshape(
                                          *dim_spatial, 1, dim_spectral)
        sp2rgb = pix_lab_rgb.sum(axis_spatial)
        one_hot_sum_reshape = one_hot_sum.reshape(num_seg,
                                                  1).repeat(dim_spectral, 1)
        sp2rgb = sp2rgb / one_hot_sum_reshape

        return sp2rgb

    def gen_hint_grid(self, image, winsize=16, num_seg=100, sigma=5):
        segments = slic(image, n_segments=num_seg, sigma=sigma, start_label=0)
        sp2rgbmean = self.gen_sp2rgbmean(image, segments)
        idx_grid = self.dominant_pool_2d(segments, winsize=winsize)
        hint_grid = sp2rgbmean[idx_grid]
        return hint_grid

    def __call__(self, x: np.ndarray):
        assert isinstance(x, np.ndarray)
        return self.gen_hint_grid(x)

        return None


if __name__ == "__main__":
    from pycomar.samples import get_img
    from torchvision.transforms import ToTensor
    img = get_img(1).resize((256, 256))
    img: torch.Tensor = ToTensor()(img)
    img = img.permute(1, 2, 0).numpy()
    sampler = HintSampler()
    k = sampler(img)
