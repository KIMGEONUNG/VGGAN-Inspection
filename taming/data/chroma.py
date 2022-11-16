from torch.utils.data import Dataset
import numpy as np
import albumentations
from PIL import Image


class ImagePaths(Dataset):

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
      self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
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
    image = self.preprocess_image(self.labels["file_path_"][i])
    example["image"] = image
    example["gray"] = self.togray(image)
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

  def __init__(self, size, training_images_list_file, use_arbitrary_gray):
    super().__init__()
    with open(training_images_list_file, "r") as f:
      paths = f.read().splitlines()
    self.data = ImagePaths(paths=paths,
                           size=size,
                           use_arbitrary_gray=use_arbitrary_gray)


class ChromaTest(ChromaBase):

  def __init__(self, size, test_images_list_file, use_arbitrary_gray):
    super().__init__()
    with open(test_images_list_file, "r") as f:
      paths = f.read().splitlines()
    self.data = ImagePaths(paths=paths,
                           size=size,
                           use_arbitrary_gray=use_arbitrary_gray)
