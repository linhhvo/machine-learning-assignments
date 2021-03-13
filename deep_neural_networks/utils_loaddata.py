#########################################################################
# Program:  utils_loaddata (untilities for the cat vs. noncat dataset)
# Source:   https://github.com/enggen/Deep-Learning-Coursera
#########################################################################
import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File("dataset/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels
    # train_set_y_orig is a 1D array (with one row, just like a list), which is different from a 2D array with only one row

    test_dataset = h5py.File("dataset/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # Transforming y arrays (1D) to 2D arrays (with only one row)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
