import random
import numpy as np
import pandas as pd
from collections import defaultdict
import time

class_num = 10
image_size = 32
img_channels = 3


def prepare_data(nexamples_train, nexamples_test, samples_to_use="pos", test_both=True):
    num_classes = 2  # Number of classes

    (train_data, train_labels), (test_data, test_labels) = load_dna_data(nexamples_train, nexamples_test, "../Data",
                                                                         ["Human"], samples_to_use, test_both=test_both)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    labelled_x = train_data[indices]
    labelled_y = train_labels[indices]

    print("======Prepare Finished======")

    labelled_y_vec = np.zeros((len(labelled_y), num_classes), dtype=np.float)
    for i, label in enumerate(labelled_y):
        labelled_y_vec[i, labelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), num_classes), dtype=np.float)
    for i, label in enumerate(test_labels):
        test_labels_vec[i, test_labels[i]] = 1.0

    return labelled_x, labelled_y_vec, test_data, test_labels_vec


def load_dna_data(num_train, num_test, base_folder, species, samples_to_use, test_both=None):
    # Input:
    #   num_train: (int) Number of training samples used, per species.
    #   num_test: (int) Number of test samples used, per species.
    #   base_folder: (string) Basefolder for data
    #   species: (array) A list of species to sample data from. This must be according to folder structure
    #   samples: (string) "neg" for only negative, "both" for both negative and positive and "pos" for only positive.
    # Output:
    #   2 tuples (x_train, y_train), (x_test, y_test):
    #       x_train, y_train: uint8 array of one-hot encoded DNA with shape (num_train, 1, 4, 500).
    #       x_test, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).
    # Example:
    #   (x_train, y_train), (x_test, y_test) = load_dna_data(10000, 5000, "Data", ["Human", "Pig", "Dolphin"], "neg")
    if samples_to_use not in ["neg", "pos", "both"]:
        raise ValueError("Samples must have values: \"neg\" (only neg), \"both\" (neg-pos) or \"pos\" (only pos). \n")

    if num_train + num_test/2 > 14000:
        raise ValueError(
            "We only have 14000 samples per species, so the sum of training and test samples cannot exceed 14000. \n")

    print("======Loading data======")
    x_train = x_test = np.empty((0, 1), dtype=int)
    y_train = y_test = np.empty((0, 1), dtype=int)

    for spec in species:
        neg_file = base_folder + '/' + spec + '/' + "negative_samples_encoded.npy"
        pos_file = base_folder + '/' + spec + '/' + "positive_samples_encoded.npy"

        with open(pos_file, 'rb') as f_pos, open(neg_file, 'rb') as f_neg:
            raw_pos_data = np.load(f_pos)
            raw_neg_data = np.load(f_neg)

            npositive = raw_pos_data.shape[0]
            print('total +ve samples' , npositive)
            nnegative = raw_neg_data.shape[0]
            print('total -ve samples', nnegative)

            pos_labels = np.ones((npositive, 1), dtype=int)
            neg_labels = np.zeros((nnegative, 1), dtype=int)

            raw_spec_data = np.concatenate([raw_pos_data, raw_neg_data])
            spec_labels = np.concatenate([pos_labels, neg_labels])

        if samples_to_use == "pos":
            # Choose num_train indices from range(0, npositive)
            train_idx = np.random.choice(npositive, num_train, replace=False)
        elif samples_to_use == "neg":
            # Choose num_train indices from range(npositive, npositive+nnegative) because positive come first
            train_idx = np.random.choice(nnegative, num_train, replace=False) + npositive
        elif samples_to_use == "both":
            train_idx = np.random.choice(nnegative + npositive, num_train*2, replace=False)

        # Always choose num_test/2 from positive and num_test/2 from negative, excluding the training
        free_pos_idx = np.setdiff1d(np.arange(npositive), train_idx)
        free_neg_idx = np.setdiff1d(np.arange(nnegative) + npositive, train_idx)
        test_pos_idx = np.random.choice(free_pos_idx, int(num_test/2))
        test_neg_idx = np.random.choice(free_neg_idx, int(num_test/2))

        test_idx = np.concatenate([test_pos_idx, test_neg_idx])
        
        x_train = np.append(x_train, raw_spec_data[train_idx])
        y_train = np.append(y_train, spec_labels[train_idx])

        x_test = np.append(x_test, raw_spec_data[test_idx])
        y_test = np.append(y_test, spec_labels[test_idx])

    # FIXME: this has hardcoded length, maybe use the ones from intialization
    SEQ_LEN = 500
    x_train = x_train.reshape((-1, SEQ_LEN, 4, 1)).transpose(0, 2, 1, 3)
    x_test = x_test.reshape((-1, SEQ_LEN, 4, 1)).transpose(0, 2, 1, 3)
    y_train = y_train.transpose()
    y_test = y_test.transpose()

    # Check for bugs
    for i in range(x_train.shape[0]):
        assert sum(sum(x_train[i, :, :, 0] == 1) == 1) == SEQ_LEN
    print("Sample training example:")
    print(x_train[0, :, :, 0])
    return (x_train, y_train), (x_test, y_test)


def one_hot_encode_string(string):
    cats = ['A', 'C', 'T', 'G']
    dummies = pd.Series(list(string)).str.get_dummies()
    return dummies.T.reindex(cats).T.fillna(0).reset_index().values[:, 1:]


def one_hot_encode(data, seq_length=500):
    one_hot_arr = np.empty((data.shape[0], seq_length, 4))
    for i, row in enumerate(data):
        one_hot_arr[i, :, :] = one_hot_encode_string(row)
    return one_hot_arr
