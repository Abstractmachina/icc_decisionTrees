import numpy as np
from numpy.random import default_rng

def read_dataset(filepath): # eg. "data/train_full.txt" 
    """Reads a given dataset.

    Args:
        filepath txt: txt file; comma del; 
        last col is dependent var which is str
        other cols are independent var which are int

    Returns:
        tuple: a tuple of 3 numpy arrays: x, y, class
    """
    x = []
    y_labels = []

    for row in open(filepath):
        if row.strip() != "": 
            row = row.strip().split(",")
            x.append(list(map(int, row[:-1]))) 
            y_labels.append(row[-1])

    classes = np.unique(y_labels) 
    x = np.array(x)
    y = np.array(y_labels)

    return (x, y, classes)

def split_dataset(x, y, test_proportion, random_generator=default_rng()):

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)