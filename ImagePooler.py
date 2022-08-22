# Note , pooling probably has more efficient and shorter functions as part of packages
# Types of pooling : Max Pooling (highest value is kept), Average Pooling

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf

def get_pools(img: np.array, pool_size: int, stride: int):
    pools = []
    for i in np.arange(img.shape[0], step=stride):
        for j in np.arange(img.shape[0], step=stride):

            # Extract the current pool
            mat = img[i:i + pool_size, j:j + pool_size]

            # Make sure it's rectangular - has the shape identical to the pool size
            if mat.shape == (pool_size, pool_size):
                # Append to the list of pools
                pools.append(mat)

    # Return all pools as a Numpy array
    return np.array(pools)


def max_pooling(pools: np.array):
    num_pools = pools.shape[0]
    # Shape of matrix after pooling = square root of the number of pools
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))

    pooled = []  # to store max values
    for pool in pools:
        pooled.append(np.max(pool))  # max pooling

    # Reshape to target shape
    return np.array(pooled).reshape(tgt_shape)


def plot_image(img: np.array):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.show()


def plot_two_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    plt.show()


conv_output = np.array([
    [10, 12,  8,  7],
    [ 4, 11,  5,  9],
    [18, 13,  7,  7],
    [ 3, 15,  2,  2]
])
#print(conv_output)
#print()

test_pools = get_pools(img = conv_output, pool_size=2, stride=2)
print(test_pools)
print()

max_pooling(pools=test_pools)
print(max_pooling(pools=test_pools))


# Image Pooling
img = Image.open("londonsquare.jpg")
img = ImageOps.grayscale(img)
img = img.resize(size=(224,224))
#plot_image(img=img)

img_pools = get_pools(img=np.array(img), pool_size=2, stride=2)
print(img_pools)
print(img_pools.shape)

img_max_pooled = max_pooling(pools=img_pools)
print(img_max_pooled)
print(img_max_pooled.shape)

#plot_two_images(img1=img, img2=img_max_pooled)


# Verification
model = tf.keras.Sequential(
    [tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
])

img_arr = np.array(img).reshape(1, 224, 224, 1)
print(img_arr.shape)

output = model.predict(img_arr).reshape(112, 112)
print(output)
print(np.array_equal(img_max_pooled, output))