import numpy as np
import tensorflow as tf

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w)] = image

    return img
