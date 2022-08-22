import os
import random
import pathlib
import shutil
import matplotlib.pyplot as plt

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
    # doesn't this only work on linux
    print(dir_str)
    return

setup_folder_stuctrure()




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