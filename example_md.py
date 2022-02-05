##############################################################################
# test program for testing limited depth decision tree
##############################################################################

import numpy as np
import collections

from classification import DecisionTreeClassifier
from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
from testing import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split
import improvedTesting_T

maxDepth = 20

if __name__ == "__main__":
    print("Loading the training dataset...")

    (x, y, classes) = read_dataset("data/train_full.txt")
    seed = 60012
    rg = default_rng(seed)

    #hyper parameter tuning
    n_folds = 5
    accuracies = np.zeros((n_folds,2))
    for i, (train_indices, val_indices, test_indices) in enumerate(improvedTesting_T.train_val_test_k_fold(n_folds, len(y), rg)):
        # set up the dataset for this fold
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_val = x[val_indices, :]
        y_val = y[val_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        # Perform grid search, i.e.
        # evaluate decision tree classifiers for K (number of levels) from 1 to 24 (inclusive)
        # and store the accuracy and classifier for each K 
        gridsearch_accuracies = []
        lastFinalDepth = 0
        for depth in range(5, 30):
            classifier = DecisionTreeClassifier()
            print(f"training decision tree {i} at depth {depth}...")
            classifier.fit(x_train, y_train, depth)
            if classifier.getFinalTreeDepth() == lastFinalDepth:
                break
            else:
                lastFinalDepth = classifier.getFinalTreeDepth()
            predictions = classifier.predict(x_val)
            acc = compute_accuracy(y_val, predictions)
            gridsearch_accuracies.append((acc, depth, classifier))

        # Select the classifier with the highest accuracy
        # and evaluate this classifier on x_test
        # key=lambda x:x[0] sorts the list by the first tuple element (the accuracy)  
        (best_acc, best_depth, best_classifier) = max(gridsearch_accuracies, key=lambda x:x[0])
        #print(gridsearch_accuracies)
        print(f"best accuracy: {best_acc}, best treeDepth: {best_depth}\n")

        predictions = best_classifier.predict(x_test)
        acc = compute_accuracy(y_test, predictions)

        #TODO: if validation accuracy and test accuracy differ too much, is it a problem? 
        #maybe should filter out based on a threshold difference.
        accuracies[i] = (acc, best_depth)
    
    print(accuracies)
    """next is to pick out the best tree depth. However, what to do if no depth occurs
    more than once? or what if the most duplicated depth has the lowest accuracy? 
    Given this problem, internal cross-validation may be necessary to provide more data.
    Computing time then is an issue in this case..."""

    # x_train, x_test, y_train, y_test = split_dataset(x, y, test_proportion=0.2, random_generator=rg)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    

    # (x_test, y_test, classes_test) = read_dataset("data/test.txt")
    # print("Training the decision tree...")
    # classifier = DecisionTreeClassifier()
    # classifier.fit(x, y, maxDepth)
    # print("Final max depth: " + str(classifier.getFinalTreeDepth()))
    # print("Loading the test set...")
    
    # print("Making predictions on the test set...")
    # predictions = classifier.predict(x_test)
    # print("\nPredictions: {}".format(predictions))
    # print("Actuals: {}".format(y_test))
    


    # seed = 60012
    # rg = default_rng(seed)

    # #evaluate output
    # print("Accuracy of prediction: ")
    # print(compute_accuracy(y_test, predictions))
    # print("\nConfusion matrix: ")
    # confusion_matrix = confusion_matrix(y_test, predictions)
    # print(confusion_matrix)
    # print("\nPrecision of prediction: ")
    # (p_random, macro_p_random) = precision(y_test, predictions)
    # print(p_random)
    # print("Macro Precision of prediction: ")
    # print(macro_p_random)
    # print("\nRecall of prediction: ")
    # (r_random, macro_r_random) = recall(y_test, predictions)
    # print(r_random)
    # print("Macro Recall of prediction")
    # print(macro_r_random)
    # (f, macro_f) = f1_score(y_test, predictions)
    # print("\nF1 score: ")
    # print(f)
    # print("\nMacro F1 score: ")
    # print(macro_f)


    # #Cross validation
    # cross_validation_acc = []
    # cross_validation_std = []
    # predictions_list = []

    # for (train_indices, test_indices) in train_test_k_fold(10, len(x), rg):
        
    #     x_train = x[train_indices]
    #     y_train = y[train_indices]
    #     x_validate = x[test_indices]
    #     y_validate = y[test_indices]

    #     classifier = DecisionTreeClassifier()
    #     classifier.fit(x_train, y_train, maxDepth)
    #     predictions = classifier.predict(x_validate)
    #     #predictions_list.append(predictions)
    #     cross_validation_acc.append(compute_accuracy(y_validate, predictions))

    #     predictions_test = classifier.predict(x_test)
    #     predictions_list.append(predictions_test)

    
    # avg_acc = sum(cross_validation_acc)/len(cross_validation_acc)
    # print("Final max depth: " + str(classifier.getFinalTreeDepth()))
    # print("\nAverage accuracy of cross validation: ")
    # print(avg_acc)
    # avg_predictions = []
    # #print(predictions_list)
    # #predictions_final

    # #0 - length of entire dataset

    # for i in range(len(predictions_list[0])):
    #     cnt = collections.Counter()
    #     for j in range(len(predictions_list)):
    #         cnt[predictions_list[j][i]] += 1
    #     #print(cnt.most_common(1))
    #     avg_predictions.append(cnt.most_common(1)[0][0])
    
    # print(avg_predictions)
    # new_avg = compute_accuracy(y_test, avg_predictions)
    # print("Avg accuracy using averaged prediction set from 10 models: ")
    # print(new_avg)