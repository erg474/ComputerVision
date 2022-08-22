# Image convolution function from scratch
# In a CNN, convolutional layers filter a number of filters (kernels) which are essentially pattern detectors
# early layers detect basic patterns such as edges, corners, etc.
# Specialised patterns detected later on, such as dog ears or cat paws, like a brain
# A single filter is a small matrix. Neural networks learn the optimal values for the filter matrix.
# A filter slider slides (convolves over set of pixels to calculate a result (like the 3D cube 1997 I made)

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def plot_image(img: np.array): # the colon ensures the variable is an np.array
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray") #try cmap='jet' viridis', 'plasma', 'inferno', 'magma', 'cividis'
    plt.show()

def plot_two_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].imshow(img1, cmap="gray")
    ax[1].imshow(img2, cmap="gray")
    plt.show()

def calculate_target_size(img_size: int, kernel_size: int):
    num_pixels = 0
    for i in range(img_size):
        # add the kernel size to the current i
        added = i + kernel_size
        #must be lower than image size
        if added <= img_size:
            num_pixels+=1
    return num_pixels

def calculate_target_size2(img_size: int, kernel_size: int):
    numpixels = img_size - kernel_size
    return numpixels

def convolve(img: np.array, kernel: np.array):
    tgt_size = calculate_target_size2(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0]
    )
    print(img.shape, img.shape[0], kernel.shape[0])
    print("tgt size: ", tgt_size)
    k = kernel.shape[0]
    print(k)

    # 2D array of zeros
    convolved_img = np.zeros(shape=(tgt_size, tgt_size))

    for i in range(tgt_size):
        for j in range(tgt_size):
            # Get current matrix value for k sized matrix convolution
            mat = img[i:i+k, j:j+k]
            #print(i,j)
            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    print(convolved_img)
    return convolved_img

def negative_to_zero(img:np.array):
    img = img.copy()
    img[img < 0] = 0
    return img

def get_padding_width_per_side(kernel_size: int):
    return kernel_size // 2

def add_padding_to_image(img: np.array, padding_width: int):
    # Padding adds a frame of black pixels the same width as the kernel size
    # Array of zeros of shape (img + padding_width)
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2, #*2 because padding needed on all sides
        img.shape[1] + padding_width * 2
    ))

    # Change the inner elements
    # e.g. if img.shape=(224,224), and img_width_padding.shape=(226,226)
    # keep the pixel wide padding on all sides, but change other values to
    # be the same as img
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img

    return img_with_padding


#Examples of different filters
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]
])

outline = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])


print(calculate_target_size(224,3))
print(calculate_target_size(224,5))

#Image Processing
img = Image.open("londonsquare.jpg")
#MUST BE SQUARE IMAGE ?
#Shouldn't be too hard to edit for rectangular image
img = ImageOps.grayscale(img)
img = img.resize(size=(224, 224))
plot_image(img=img)
#plt.show()


imagenumpyarray = np.array(img)

img_sharpen = convolve(img=imagenumpyarray, kernel=sharpen)
img_blurred = convolve(img=imagenumpyarray, kernel=blur)
img_outlined =  convolve(img=imagenumpyarray, kernel = outline)

kernel_plots = '''plot_two_images(
    img1 = img,
    img2 = negative_to_zero(img_sharpen)
)

plot_two_images(
    img1 = img,
    img2 = img_blurred
)

plot_two_images(
    img1 = img,
    img2 = negative_to_zero(img_outlined)
)
'''


# Resulting images are 222x222, Convolution Padding is needed to get to 224x224
#odo: No padding on top of image?
img_with_padding_3x3 = add_padding_to_image(
    img = np.array(img),
    padding_width = get_padding_width_per_side(kernel_size=3)
)

print(img_with_padding_3x3.shape)
plot_image(img_with_padding_3x3)
plt.show()
#the resultant matrix is outlined by 0 values


img_with_padding_5x5 = add_padding_to_image(
    img=np.array(img),
    padding_width=get_padding_width_per_side(kernel_size=5)
)

print(img_with_padding_5x5.shape)
plot_image(img_with_padding_5x5)

print(img_with_padding_3x3)
print(img_with_padding_5x5)

img_padded_3x3_sharpened = convolve(img=img_with_padding_3x3, kernel=sharpen)
print(img_padded_3x3_sharpened.shape)

plot_two_images(
    img1=img,
    img2=img_padded_3x3_sharpened
)