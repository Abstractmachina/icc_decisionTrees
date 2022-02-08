##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import collections
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split
import math
from random_forest import RandomForestClassifier


# TODO: get rid of Y TEST WHEN DONE!!!
def train_and_predict(x_train, y_train, x_test, y_test):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################
       

    # TODO: Train new classifier
    random_forest = RandomForestClassifier(100, int(math.sqrt(len(x_train[0,:]))), 0.67)
    models = random_forest.run_forest(x_train, y_train, x_test)
    print("\nTotal Number of Models: ")
    print(len(models))

    #new list of predictions
    predictions_list = []
    all_models_accuracy = np.zeros((100,),dtype=float)

    #run every model through predictions
    for i, model in enumerate(models):
        predictions_test = random_forest.improved_predict(x_test, model)
        predictions_list.append(predictions_test)
        all_models_accuracy[i] = compute_accuracy(y_test, predictions_test)
    print("\nAccuracyf of each model from the tree: ")
    print(all_models_accuracy)

    #store the average predictions
    avg_predictions = np.zeros((x_test.shape[0],), dtype=np.object)

    #count and return most commonly occuring label
    for i in range(len(predictions_list[0])):
        cnt = collections.Counter()
        for j in range(len(predictions_list)):
            cnt[predictions_list[j][i]] += 1
        avg_predictions[i] = cnt.most_common(1)[0][0]

    return avg_predictions

    #return avg_predictions