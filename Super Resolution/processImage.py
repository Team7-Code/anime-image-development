import numpy as np

def normalizeImages(images):
    images = images.astype('float32')
    images = np.divide(images, 255)
    return images

def denormalizeImages(images):
    images = images*255
    return images.astype(np.uint8)

def denormalize_HR(images):
    images = (images*127.5) + 127.5
    return images.astype(np.uint8)