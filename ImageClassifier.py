# A combination of convolutions and pooling to classify images
# ANNs are terrible for this, but CNNs are good

# Remember need to remove a couple of images, one was 666

import os
import pathlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(42)

from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore")

def visualize_batch(batch: tf.keras.preprocessing.image.DirectoryIterator):
    n = 64
    num_row, num_col = 8, 8
    fig, axes = plt.subplots(num_row, num_col, figsize=(3 * num_col, 3 * num_row))

    for i in range(n):
        img = np.array(batch[0][i] * 255, dtype='uint8')
        ax = axes[i // num_col, i % num_col]
        ax.imshow(img)

    plt.tight_layout()
    plt.show()

img1 = Image.open('data/train/cat/1.jpg')
print(np.array(img1).shape)

img2 = Image.open('data/train/dog/0.jpg')
print(np.array(img2).shape)

# TensorFlow Data Loaders
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

train_data = train_datagen.flow_from_directory(
    directory="data/train/",
    target_size=(224, 224), # size all images will be resized to
    class_mode="categorical", # categorical: distinct image classification
    batch_size=64, # images shown to neural network
    seed=42
)

first_batch = train_data.next() # 1 batch = 64 images
print(first_batch[0].shape, first_batch[1].shape)

#visualize_batch(batch=first_batch)

# Train CNN
train_data = train_datagen.flow_from_directory(
    directory="data/train/",
    target_size=(224,224),
    class_mode="categorical",
    batch_size=64,
    seed=42
)

valid_datagen = valid_datagen.flow_from_directory(
    directory="data/validation/",
    target_size=(224,224),
    class_mode="categorical",
    batch_size=64,
    seed=42
)