##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np
import collections

from classification import DecisionTreeClassifier
from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
from testing import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split

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
    #seed = 60012
    #rg = default_rng(seed)
    #x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)
    #print(x_train)
    #print(y_train)
    #print(x_test)
    #print(y_test)
    (x_test, y_test, classes_test) = read_dataset("data/test.txt")
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)

    print("Loading the test set...")
    
    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("\nPredictions: {}".format(predictions))
    print("Actuals: {}".format(y_test))
    

    '''               
    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))'''

    seed = 60012
    rg = default_rng(seed)

    #evaluate output
    print("Accuracy of prediction: ")
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
    #Cross validation
    cross_validation_acc = []
    cross_validation_std = []
    predictions_list = []

    for (train_indices, test_indices) in train_test_k_fold(10, len(x), rg):
        
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_validate = x[test_indices]
        y_validate = y[test_indices]

        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_validate)
        #predictions_list.append(predictions)
        cross_validation_acc.append(compute_accuracy(y_validate, predictions))

        predictions_test = classifier.predict(x_test)
        predictions_list.append(predictions_test)

    
    avg_acc = sum(cross_validation_acc)/len(cross_validation_acc)
    print("\nAverage accuracy of cross validation: ")
    print(avg_acc)
    avg_predictions = []
    #print(predictions_list)
    #predictions_final

    #0 - length of entire dataset

    for i in range(len(predictions_list[0])):
        cnt = collections.Counter()
        for j in range(len(predictions_list)):
            cnt[predictions_list[j][i]] += 1
        #print(cnt.most_common(1))
        avg_predictions.append(cnt.most_common(1)[0][0])
    
    print(avg_predictions)
    new_avg = compute_accuracy(y_test, avg_predictions)
    print("Avg accuracy using averaged prediction set from 10 models: ")
    print(new_avg)












    