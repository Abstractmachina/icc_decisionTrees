##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import json
import collections
from posixpath import split
from classification import DecisionTreeClassifier
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split
import testing
import math



def train_and_predict(x_train, y_train, x_test):
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

    #p value used to splitting features
    p_value = int(math.sqrt(len(x_train[0,:])))
    #p_value = 6

    #random seed
    seed = 60012
    rg = default_rng(seed)

    #folds parameter
    n_folds = 100

    #lists to store outputs
    cross_validation_acc = []
    cross_validation_std = []
    predictions_list = []

    for (train_indices, test_indices) in train_test_k_fold(n_folds, len(x_train), 0.67, rg):
        x_forest = x_train[train_indices]
        y_forest = y_train[train_indices]
        x_validate = x_train[test_indices]
        y_validate = y_train[test_indices]
        
        #train tree using random forest classifier
        improved_fit(x_forest, y_forest, p_value)

        #predict using the tree on validate set
        predictions = improved_predict(x_validate)
        cross_validation_acc.append(compute_accuracy(y_validate, predictions))

        #then run predict with test data
        predictions_test = improved_predict(x_test)
        predictions_list.append(predictions_test)
        #print("Done with improve loop")

    # set up an empty (M, ) numpy array to store the predicted labels 
    # feel free to change this if needed
    avg_acc = sum(cross_validation_acc)/len(cross_validation_acc)
    print()
    print("\nAverage accuracy of cross validation: ")
    print(avg_acc)
    avg_predictions = np.zeros((x_test.shape[0],), dtype=np.object)

    for i in range(len(predictions_list[0])):
        cnt = collections.Counter()
        for j in range(len(predictions_list)):
            cnt[predictions_list[j][i]] += 1
        avg_predictions[i] = cnt.most_common(1)[0][0]

    return avg_predictions

def improved_fit(x, y, p_value):
    classes = np.unique(y)
        #print("Entropy for the set is: ")
        #print(testing.calculate_entrophy(x, y, classes))
    model = {}


    testing.random_forest_classifier(x, y, classes, 0, model, p_value)
    
    # write model to file
    
    with open('model.json', 'w') as f:
        f.write(json.dumps(model))
    #print("done with fitting")

def improved_predict(x):
    """ Predicts a set of samples using the trained DecisionTreeClassifier.
    
    Assumes that the DecisionTreeClassifier has already been trained.
    
    Args:
    x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                        M is the number of test instances
                        K is the number of attributes
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                    class label for each instance in x
    """
    # set up an empty (M, ) numpy array to store the predicted labels 
    # feel free to change this if needed
    predictions = np.zeros((x.shape[0],), dtype=np.object)
    
    
    #######################################################################
    #                 ** TASK 2.2: COMPLETE THIS METHOD **
    #######################################################################
    
    # read model from file
    with open('model.json', 'r') as f:
        model = json.load(f)

    # TODO guarantee there's a better way to do this with numpy but we can look into that in future
    for row_number in range(0, len(x)):
        check_nodes(x, model, predictions, row_number)
    
    #print("done with predicting")
    return predictions
    
def check_nodes(x, model, predictions, row_number):
    while True:
        # loop through every key at this level of the model to see which is viable
        k = model.keys()
        #print(k)
        for key in k:
            # base case, if we reach a terminating node then set predictions[row_number] to v
            if (key == "terminating_node"):
                predictions[row_number] = model[key]
                return

            # split the key out into its constituent parts
            split_key = key.split(',')
            feature_index = int(split_key[0])
            value = int(split_key[1])
            #print(feature_index, value)
            #do it all again from the next node, recursively.
            if (x[row_number, feature_index] >= value):
                model = model[key]
                break    