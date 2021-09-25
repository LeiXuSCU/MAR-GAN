import os

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path):
    images = []
    assert os.path.isdir(path), '%s is not a valid directory' % path
    for root, _, file_names in sorted(os.walk(path)):
        for file_name in file_names:
            if is_image_file(file_name):
                path = os.path.join(root, file_name)
                images.append(path)
    return images


def get_transform():
    transform_list = [
        transforms.Grayscale(1),  # grayscale
        transforms.Resize([512, 512], transforms.InterpolationMode.BICUBIC),  # resize
        # crop  random.randint(0, np.maximum(0, resize - crop_size))
        transforms.Lambda(lambda img: __crop(img, (0, 0), 512)),
        transforms.Lambda(lambda img: __flip(img, random.random() > 0.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def tensor2im(input_image, im_type=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(im_type)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)
