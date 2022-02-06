##############################################################################
# test program for testing limited depth decision tree
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from numpy.random import default_rng
from testing import compute_accuracy, confusion_matrix, precision, recall, f1_score
import improvedTesting_T

maxDepth = 20
n_folds = 10

if __name__ == "__main__":
    print("Loading the training dataset...")

    (x, y, classes) = read_dataset("data/train_full.txt")
    seed = 60012
    rg = default_rng(seed)
    
    #____________   hyper parameter tuning  ______________________
    print("Starting cross-validation...")
    
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
            print(f"Cross-validating decision tree {i} at depth {depth}...")
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
    print("Cross-validation results:")
    print("[Accuracy , Depth , Classifier]")
    print(accuracies) 
    """next is to pick out the best tree depth. However, what to do if no depth 
    occurs more than once? or what if the most duplicated depth has the lowest 
    accuracy? Given this problem, internal cross-validation may be necessary to 
    provide more data. Computing time then is an issue in this case..."""
    # #test array
    # accuracies = np.array([
    #     [0.88205128, 12],
    #     [0.89230769, 15],
    #     [0.90128205, 11],
    #     [0.92564103, 18],
    #     [0.90512821, 12]
    # ])

    uniqueLen = len(np.unique(accuracies[:,1]))
    depthCount = np.stack([
        np.unique(accuracies[:,1]), 
        np.zeros((uniqueLen,)),
        np.zeros((uniqueLen,))], 
        axis = 1
    )


    print("Tuning hyperparameter ...")

    #count majority depth
    for a in accuracies:
        print((a[0], a[1]))
        for d in depthCount:
            if a[1] == d[0]:
                d[1] += 1
                d[2] += a[0]
                break
    result = []
    #if there is majority
    if all(depthCount[:,1] == 1):
        result = max(depthCount, key=lambda x:x[2])
        
    #if there is no majority, take highest accuracy depth
    else:
        result = max(depthCount, key=lambda x:x[1])
    #print((depth, count, accuracy))
    result[2] = result[2] / result[1]
    print(result)
    hyperDepth = result[0]

    
    print(f"The optimum depth is {hyperDepth}!\n")


    #___________________    train with hyperparameter   __________________
    print("************* ...Training model ... ******************************")
    print("Loading datasets...")
    (x_train, y_train, classes_train) = read_dataset("data/train_full.txt")
    (x_test, y_test, classes_test) = read_dataset("data/test.txt")
    print("Training decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y, hyperDepth)

    print("Making predictions on test set...")
    predictions = classifier.predict(x_test)
    print("\nPredictions: {}".format(predictions))
    print("Actuals: {}".format(y_test))


    #___________________  evaluate output    ___________________________
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