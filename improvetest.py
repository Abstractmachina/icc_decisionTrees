import numpy as np
import collections

from classification import DecisionTreeClassifier
from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split

if __name__ == "__main__":
    print("Loading the training dataset...")
    '''x_train = np.array([
            [5,7,1],
            [4,6,2],
            [4,6,3], 
            [1,3,1], 
            [2,1,2], 
            [5,2,6]
        ])
    
    y_train = np.array(["A", "A", "A", "C", "C", "C"])'''
    
    (x, y, classes) = read_dataset("data/train_full.txt")

    (x_test, y_test, classes_test) = read_dataset("data/test.txt")

    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, y_test)
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