# Sorts data directory based on classes and randomly sorts images into train/test/validation

import os
import random
import pathlib
import shutil
import matplotlib.pyplot as plt
import time
from progress.bar import Bar


def setup_folder_stuctrure():
    # Create base folders if they don't exist
    if not dir_data.exists():  dir_data.mkdir()
    if not dir_train.exists(): dir_train.mkdir()
    if not dir_valid.exists(): dir_valid.mkdir()
    if not dir_test.exists():  dir_test.mkdir()

    # Create subfolders for each class
    for cls in img_classes:
        if not dir_train.joinpath(cls).exists(): dir_train.joinpath(cls).mkdir()
        if not dir_valid.joinpath(cls).exists(): dir_valid.joinpath(cls).mkdir()
        if not dir_test.joinpath(cls).exists():  dir_test.joinpath(cls).mkdir()

    # Print the directory structure
    # Credits - https://stackoverflow.com/questions/3455625/linux-command-to-print-directory-structure-in-the-form-of-a-tree
    dir_str = os.system('''ls -R data | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/' ''')
    # won't output properly on windows
    print(dir_str)
    return

def train_test_validation_split(src_folder: pathlib.PosixPath, class_name: str):
    # 80:10:10 split for train, test, validation
    # use random number to put items into categories

    # For tracking
    n_train, n_valid, n_test = 0, 0, 0

    # Random seed for reproducibility
    random.seed(42)
    count=0

    # Scrollbar
    bar = Bar("Processing", max=12501)

    # Iterate over every image
    for file in src_folder.iterdir():
        img_name = str(file).split("\\")[-1]
        #NOTE : ON WINDOWS SPLIT BY "\\", LINUX: "/"


        # Make sure it's JPG
        if file.suffix == ".jpg":

            x = random.random()

            # Choosing where image goes in folder
            tgt_dir = ''

            # .80 or below
            if x <= pct_train:
                tgt_dir = 'train'
                n_train += 1

            # Between .80 and .90
            elif pct_train < x <= (pct_train + pct_valid):
                tgt_dir = 'validation'
                n_valid += 1

            # Above .90
            else:
                tgt_dir = 'test'
                n_test += 1

            # Copy the image
            shutil.copy(
                src=file,
                # data/<train|valid|test>/<cat\dog>/<something>.jpg
                dst=f'{str(dir_data)}\\{tgt_dir}\\{class_name}\\{img_name}'
            )

        count+=1
        bar.next()
        #if count % 100 == 0:
            #print(count, "/", len(os.listdir(src_folder)))

    bar.finish()
    return {
        'source': str(src_folder),
        'target': str(dir_data),
        'n_train': n_train,
        'n_validaiton': n_valid,
        'n_test': n_test
    }

def plot_random_sample(img_dir: pathlib.PosixPath):
    # How many images we're showing
    n = 10
    # Get absolute paths to these N images
    imgs = random.sample(list(img_dir.iterdir()), n)

    # Make sure num_row * num_col = n
    num_row = 2
    num_col = 5

    # Create a figure
    fig, axes = plt.subplots(num_row, num_col, figsize=(3.5 * num_col, 3 * num_row))
    # For every image
    for i in range(num_row * num_col):
        # Read the image
        img = plt.imread(str(imgs[i]))
        # Display the image
        ax = axes[i // num_col, i % num_col]
        ax.imshow(img)
        # Set title as <train|test|validation>/<cat\dog>/<img_name>.jpg
        ax.set_title('/'.join(str(imgs[i]).split('/')[-3:]))

    plt.tight_layout()
    plt.show()


# Distinct image classes
img_classes = ["cat", "dog"]

# Folders for training, testing and validation subsets
dir_data = pathlib.Path.cwd().joinpath("data")
dir_train = dir_data.joinpath("train")
dir_valid = dir_data.joinpath("validation")
dir_test = dir_data.joinpath("test")

# Train/Test/Validation split config
pct_train = 0.8
pct_valid = 0.1
pct_test = 0.1

setup_folder_stuctrure()

print("Pathlib Woah", pathlib.Path.cwd().joinpath("PetImages/Cat"))

start=time.time()
print(start)

train_test_validation_split(
    src_folder=pathlib.Path.cwd().joinpath("PetImages/Cat"),
    class_name="cat"
)
end=time.time()
print(end-start)


start=time.time()
print(start)
train_test_validation_split(
    src_folder=pathlib.Path.cwd().joinpath("PetImages/Dog"),
    class_name="dog"
)
print(time.time()-start)

print("Finished")
plot_random_sample(img_dir=pathlib.Path().cwd().joinpath('data/train/cat'))
