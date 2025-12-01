import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

try:
    import cv2
except ImportError:
    pass


def get_sketch(image):
    edges = cv2.Canny(image=image, threshold1=20, threshold2=180)
    edges = cv2.GaussianBlur(edges, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.bitwise_not(edges)
    edges[edges < 255] = 0
    return edges


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_P_from_transform_matrix(matrix):

    location = matrix[0:3, 3]

    rotation = np.transpose(matrix[0:3, 0:3])
    t = np.tan(np.pi / 6.0)
    width = 224

    return np.array([[112 / t, 0, width/2], [0, 112 / t, width/2], [0, 0, 1]]) \
        @ np.concatenate([rotation, np.expand_dims(-1*rotation @ location, 1)], 1)
