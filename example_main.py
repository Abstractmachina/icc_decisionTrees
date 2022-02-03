##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
from testing import compute_accuracy

if __name__ == "__main__":
<<<<<<< HEAD
    print("Loading the training dataset...");
    x = np.array([
=======
    print("Loading the training dataset...")
    '''x_train = np.array([
>>>>>>> fc969b6cd7bee241756b46bbf1e9bce8d8104f07
            [5,7,1],
            [4,6,2],
            [4,6,3], 
            [1,3,1], 
            [2,1,2], 
            [5,2,6]
        ])
    
<<<<<<< HEAD
    y = np.array(["A", "A", "A", "C", "C", "C"])
    
    (x, y, classes) = read_dataset("data/train_full.txt")

=======
    y_train = np.array(["A", "A", "A", "C", "C", "C"])'''
    
    (x, y, classes) = read_dataset("data/train_full.txt")
    seed = 60012
    rg = default_rng(seed)
    x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)
    #print(x_train)
    #print(y_train)
    #print(x_test)
    #print(y_test)
>>>>>>> fc969b6cd7bee241756b46bbf1e9bce8d8104f07
    
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

    print("Loading the test set...")
    
    '''x_test = np.array([
                [1,6,3], 
                [0,5,5], 
                [1,5,0], 
                [2,4,2]
            ])
    
    y_test = np.array(["A", "A", "C", "C"])'''
    
    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("\nPredictions: {}".format(predictions))
    print("Actuals: {}".format(y_test))
    
    '''x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])
                   
    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))'''
    
    #evaluate output
    print("Accuracy of prediction: ")
    print(compute_accuracy(y_test, predictions))