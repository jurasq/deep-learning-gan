import random
import numpy as np
import pandas as pd
from collections import defaultdict
import time

class_num = 10
image_size = 32
img_channels = 3

def prepare_data(n):
    num_classes = 2 #Number of classes

    (train_data, train_labels), (test_data, test_labels) = load_dna_data(100, 50, "../Data", ["Human"], 1)

    #Pre-processing should happen here if wanted if wanted. We have processed data.

    criteria = n//num_classes
    input_dict, labelled_x, labelled_y, unlabelled_x, unlabelled_y = defaultdict(int), list(), list(), list(), list()

    for image, label in zip(train_data,train_labels) :
        if input_dict[int(label)] != criteria :
            input_dict[int(label)] += 1
            labelled_x.append(image)
            labelled_y.append(label)

        unlabelled_x.append(image)
        unlabelled_y.append(label)

    labelled_x = np.asarray(labelled_x)
    labelled_y = np.asarray(labelled_y)
    unlabelled_x = np.asarray(unlabelled_x)
    unlabelled_y = np.asarray(unlabelled_y)

    print("labelled data:", np.shape(labelled_x), np.shape(labelled_y))
    print("unlabelled data :", np.shape(unlabelled_x), np.shape(unlabelled_y))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(labelled_x))
    labelled_x = labelled_x[indices]
    labelled_y = labelled_y[indices]

    indices = np.random.permutation(len(unlabelled_x))
    unlabelled_x = unlabelled_x[indices]
    unlabelled_y = unlabelled_y[indices]

    print("======Prepare Finished======")

    labelled_y_vec = np.zeros((len(labelled_y), num_classes), dtype=np.float)
    for i, label in enumerate(labelled_y) :
        labelled_y_vec[i, labelled_y[i]] = 1.0

    unlabelled_y_vec = np.zeros((len(unlabelled_y), num_classes), dtype=np.float)
    for i, label in enumerate(unlabelled_y) :
        unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), num_classes), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0

    return labelled_x, labelled_y_vec, unlabelled_x, unlabelled_y_vec, test_data, test_labels_vec


def load_dna_data(num_train, num_test, base_folder, species, samples):
    #Input:
    #   num_train: (int) Number of training samples used, per species.
    #   num_test: (int) Number of test samples used, per species.
    #   base_folder: (string) Basefolder for data
    #   species: (array) A list of species to sample data from. This must be according to folder structure
    #   samples: (int) -1 for only negative, 0 for both negative and positive and 1 for only positive.
    #Output:
    #   2 tuples (x_train, y_train), (x_test, y_test):
    #       x_train, y_train: uint8 array of one-hot encoded DNA with shape (num_train, 1, 4, 500).
    #       x_test, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).
    #Example:
    #   (x_train, y_train), (x_test, y_test) = load_dna_data(10000, 5000, "Data", ["Human", "Pig", "Dolphin"], -1)
    if not samples in [-1, 0, 1]:
        raise ValueError("Samples must have values: -1 (only neg), 0 (neg-pos) or 1 (only pos). \n")

    if num_train + num_test > 14000:
        raise ValueError("We only have 14000 samples per species, so the sum of training and test samples cannot exceed 14000. \n")

    print("======Loading data======")
    raw_train_data = raw_test_data = np.empty((0, 1),dtype=str)
    labels_train = labels_test = np.empty((0, 1),dtype=int)

    for spec in species:
        neg_file = base_folder + '/' + spec + '/' + "negative_samples"
        pos_file = base_folder + '/' + spec + '/' + "positive_samples"
        if samples == -1:
            with open(neg_file) as f:
                raw_spec_data = np.loadtxt(f,dtype=str)
                spec_labels = np.zeros((raw_spec_data.shape[0], 1),dtype=int)
        elif samples == 1:
            with open(pos_file) as f:
                raw_spec_data = np.loadtxt(f,dtype=str)
                spec_labels = np.ones((raw_spec_data.shape[0], 1),dtype=int)
        else:
            with open(pos_file) as f_pos, open(neg_file) as f_neg:
                raw_pos_data = np.loadtxt(f_pos,dtype=str)
                raw_neg_data = np.loadtxt(f_neg,dtype=str)

                pos_labels = np.one((raw_pos_data.shape[0], 1),dtype=int)
                neg_labels = np.zeros((raw_neg_data.shape[0], 1),dtype=int)

                raw_spec_data = np.concatenate(raw_pos_data, raw_neg_data)
                spec_labels = np.concatenate(pos_labels, neg_labels)
        train_idx = np.random.choice(num_train+num_test, num_train, replace=False)
        test_idx = np.setxor1d(np.arange(num_train+num_test), train_idx)

        raw_train_data = np.append(raw_train_data, raw_spec_data[train_idx])
        labels_train = np.append(labels_train, spec_labels[train_idx])

        raw_test_data = np.append(raw_test_data, raw_spec_data[test_idx])
        labels_test = np.append(labels_test, spec_labels[test_idx])

    print("======Encoding data======")
    x_train = one_hot_encode(raw_train_data)
    y_train = labels_train[:, np.newaxis]

    x_test = one_hot_encode(raw_test_data)
    y_test = labels_test[:, np.newaxis]
    print("======Encoding finished======")

    return (x_train, y_train), (x_test, y_test)

def one_hot_encode_string(string):
    return pd.Series(list(string)).str.get_dummies().reset_index().values[:,1:]

def one_hot_encode(data, seq_length=500):
    one_hot_arr = np.empty((data.shape[0], seq_length, 4))
    for i, row in enumerate(data):
        one_hot_arr[i,:,:] = one_hot_encode_string(row)
    return one_hot_arr
