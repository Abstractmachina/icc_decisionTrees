##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np
import collections

from classification import DecisionTreeClassifier
from read_data import read_dataset
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold

if __name__ == "__main__":
    print("Loading the training dataset...")
    (x, y, classes) = read_dataset("data/train_full.txt")


    (x_test, y_test, classes_test) = read_dataset("data/test.txt")
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)

    print("Loading the test set...")

    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("\nPredictions: {}".format(predictions))
    print("Actuals: {}".format(y_test))
    

    seed = 60012
    rg = default_rng(seed)

    ################### Evaluate output ###################
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


    ################ Cross validation #####################
    n_folds = 10
    cross_validation_acc = np.zeros((n_folds, ), dtype=float)
    predictions_list = []

    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(x), 0.3, rg)):
        
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_validate = x[test_indices]
        y_validate = y[test_indices]

        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_validate)
        cross_validation_acc[i] = compute_accuracy(y_validate, predictions)

        predictions_test = classifier.predict(x_test)
        predictions_list.append(predictions_test)

    print("\nAverage accuracy of cross validation: ")
    print(cross_validation_acc.mean())
    print("\nStandard Deviation: ")
    print(cross_validation_acc.std())
    avg_predictions = []

    for i in range(len(predictions_list[0])):
        cnt = collections.Counter()
        for j in range(len(predictions_list)):
            cnt[predictions_list[j][i]] += 1
        avg_predictions.append(cnt.most_common(1)[0][0])
    
    print("\nModal predictions from ", n_folds, " cross validations: ")
    print(avg_predictions)
    new_avg = compute_accuracy(y_test, avg_predictions)
    print("Avg accuracy using averaged prediction set from 10 models: ")
    print(new_avg)

    '''               
    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))'''












    