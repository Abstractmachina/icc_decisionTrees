import numpy as np
import collections

from classification import DecisionTreeClassifier
from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split

if __name__ == "__main__":
    print("Loading the training dataset...")

    (x, y, classes) = read_dataset("data/train_full.txt")

    (x_test, y_test, classes_test) = read_dataset("data/test.txt")

    # Generate a validation set
    # 0.20 reserved for validation. must take 0.4125 of remaining test set to train tree on 33% bootstrap data
    seed = 60012
    rg = default_rng(seed)
    x_train, x_validate, y_train, y_validate = split_dataset(x, y, 0.2, rg)

    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x_train, y_train, x_test, y_test, x_validate, y_validate)
    print("Predictions: {}".format(predictions))

    print("\nAccuracy of prediction: ")
    print(compute_accuracy(y_test, predictions))

    print("\nConfusion matrix: ")
    confusion_matrix = confusion_matrix(y_test, predictions)
    print(confusion_matrix)
    print("\nPrecision of prediction: ")
    (p_random, macro_p_random) = precision(y_test, predictions)
    print(p_random)
    print("Macro Precision of prediction: ")
    print(macro_p_random)
    print("\nRecall of prediction: ")
    (r_random, macro_r_random) = recall(y_test, predictions)
    print(r_random)
    print("Macro Recall of prediction")
    print(macro_r_random)
    (f, macro_f) = f1_score(y_test, predictions)
    print("\nF1 score: ")
    print(f)
    print("\nMacro F1 score: ")
    print(macro_f)