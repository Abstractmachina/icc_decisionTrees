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
from evaluate import compute_accuracy
import math
from random_forest import RandomForestClassifier

def train_and_predict(x_train, y_train, x_test, x_val, y_val, y_test=None):

    assert x_train.shape[0] == len(y_train) and x_test.shape[0] == len(y_test) and x_val.shape[0] == len(y_val), \
        "Training failed. x and y must have the same number of instances."

    # Initialise new random forest classifier class
    p_value = int(math.sqrt(len(x_train[0,:])))
    total_trees = 125
    data_prop = 0.5
    random_forest = RandomForestClassifier(total_trees, p_value, data_prop)

    # run classifier: Random forest classifier object stores every tree generated in a list
    random_forest.run_forest(x_train, y_train, x_test)

    # new list of predictions
    predictions_list = []
    all_models_accuracy = np.zeros((total_trees,),dtype=float)

    # run every model in the classifier's tree list, prune first, and then add prediction 
    # predictions list
    for i, model in enumerate(random_forest.models):
        # We use our validation set that is completely distinct from the data we used to
        # train the models in our random forest. We prune based on model performance on
        # this validation set

        random_forest.prune_nodes(x_val, y_val, model)

        predictions_test = random_forest.improved_predict(x_test, model)
        predictions_list.append(predictions_test)
        all_models_accuracy[i] = compute_accuracy(y_test, predictions_test)

    # Store the most commonly occuring prediction into new prediction array
    avg_predictions = np.zeros((x_test.shape[0],), dtype=np.object)

    # count and return most commonly occuring label
    for i in range(len(predictions_list[0])):
        cnt = collections.Counter()
        for j in range(len(predictions_list)):
            cnt[predictions_list[j][i]] += 1
        avg_predictions[i] = cnt.most_common(1)[0][0]

    return avg_predictions

