import numpy as np
import torch
""" A custom ToTensor class, for use with 16bit grayscale pngs. As PIL is incompatible with this image type,
    we made our own.
"""
class ToTensor3D(object):
    """Convert ndarrays in sample to Tensors.
       Returns a (3,Y,X) tensor from a (Y,X) ndarray
    """
    def __call__(self, image):
        new_shape = (3,) + image.shape
        dup_img = np.broadcast_to(image, new_shape)
        return torch.from_numpy(dup_img.copy())
