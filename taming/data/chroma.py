from torch.utils.data import Dataset
import numpy as np
import albumentations
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import skimage.measure
import random
from ..algorithms import HintSampler, GrayConversion


class ImagePaths(Dataset):

    def __init__(self,
                 paths,
                 size=None,
                 random_crop=False,
                 labels=None,
                 togray="classic"):
        self.size = size
        self.random_crop = random_crop
        self.togray_method = togray

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.hint_sampler = HintSampler()

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,
                                                         width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,
                                                         width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

        self.togray = GrayConversion(
            name=self.togray_method,
            preprocess=lambda x: (x + 1) * 0.5,
            postprocess=lambda x: (x * 2) - 1,
        )

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def gen_mask(self, prop=0.5, spatial=(16, 16)):
        mask = np.random.binomial(1, p=prop, size=spatial)
        return mask

    def __getitem__(self, i):
        example = dict()
        # The image range is [-1, 1]
        image = self.preprocess_image(self.labels["file_path_"][i])
        example["image"] = image
        example["gray"] = self.togray(image)
        example["hint"] = self.hint_sampler(image)
        example["mask"] = self.gen_mask()
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class ChromaBase(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class ChromaTrain(ChromaBase):

    def __init__(self, size, training_images_list_file, togray):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths,
                               size=size,
                               togray=togray)


class ChromaTest(ChromaBase):

    def __init__(self, size, test_images_list_file, togray):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths,
                               size=size,
                               togray=togray)


class ImagePaths2(Dataset):

    def __init__(self,
                 paths,
                 size=None,
                 random_crop=False,
                 labels=None,
                 use_arbitrary_gray=False):
        self.size = size
        self.random_crop = random_crop
        self.use_arbitrary_gray = use_arbitrary_gray

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,
                                                         width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,
                                                         width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

        if self.use_arbitrary_gray:
            self.togray = self.convert2gray
        else:
            raise NotImplementedError()

    def __len__(self):
        return self._length

    def convert2gray(self, x):
        w = np.random.randn(3)
        x = (x @ w)[..., None]
        x = np.tanh(x).astype(np.float32)
        return x

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        path_img = self.labels["file_path_"][i]
        image = self.preprocess_image(path_img)  # resize + crop + rerange
        example["image"] = image
        example["gray"] = self.togray(image)
        example["hint_grid"] = ImagePaths2.gen_hint_grid(image)
        example["mask"] = ImagePaths2.gen_mask_hint_label()[..., None]

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

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

    @staticmethod
    def dominant_pool_2d(spatial: np.ndarray, winsize=16):
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

    @staticmethod
    def gen_sp2rgbmean(image: np.ndarray, segments: np.ndarray):
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

    @staticmethod
    def gen_hint_grid(image, winsize=16, num_seg=100, sigma=5):
        segments = slic(image, n_segments=num_seg, sigma=sigma, start_label=0)
        sp2rgbmean = ImagePaths2.gen_sp2rgbmean(image, segments)
        idx_grid = ImagePaths2.dominant_pool_2d(segments, winsize=winsize)
        hint_grid = sp2rgbmean[idx_grid]
        return hint_grid


class ChromaTrain2(ChromaBase):

    def __init__(self, size, training_images_list_file, use_arbitrary_gray):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths2(paths=paths,
                                size=size,
                                use_arbitrary_gray=use_arbitrary_gray)


class ChromaTest2(ChromaBase):

    def __init__(self, size, test_images_list_file, use_arbitrary_gray):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths2(paths=paths,
                                size=size,
                                use_arbitrary_gray=use_arbitrary_gray)
