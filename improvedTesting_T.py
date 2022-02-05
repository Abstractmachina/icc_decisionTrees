from typing import final
from xml.dom.minicompat import NodeList
import numpy as np
from read_data import read_dataset
import math
from numpy.random import default_rng
from testing import calculate_best_info_gain, split_by_best_rule, k_fold_split


### Recursion with max depth:
def induce_tree_maxDepth(x, y, classes, node_level, parent_node, maxDepth, final_depth):

    if (len(y) == 0):
        return
    # base case
    #if maxdepth is 0, will be ignored
    if node_level+1 == maxDepth and maxDepth != 0:
        #find majority label and set as terminating node.
        labelCount = {}
        for l in np.unique(y):
            labelCount[l] = 0
        for label in y:
            labelCount[label] += 1
        maxLabel = max(labelCount, key= labelCount.get)
        parent_node["terminating_node"] = maxLabel
        if node_level+1 > final_depth:
            final_depth = node_level+1
        return final_depth
    if len(np.unique(y)) == 1:
        parent_node["terminating_node"] = y[0]
        if node_level+1 > final_depth:
            final_depth = node_level+1
        return final_depth
    if len(np.unique(x, axis=0)) <= 1:
        unique, frequency = np.unique(y, return_counts=True) #unique array with corresponding count
        parent_node["terminating_node"] = unique[np.argmax(frequency)] # place value into terminating
        #print(unique[np.argmax(frequency)])
        if node_level+1 > final_depth:
            final_depth = node_level+1
        return final_depth
    
    (feature_index, split_value) = calculate_best_info_gain(x, y, classes)

    child_node_left = {}
    parent_node[str(feature_index) + ',' + str(split_value)] = child_node_left

    child_node_right = {}
    parent_node[str(feature_index) + ',' + str(0)] = child_node_right

    (left_x, left_y, right_x, right_y) = split_by_best_rule(feature_index, split_value, x, y)

    d1 = induce_tree_maxDepth(left_x, left_y, classes, node_level+1, child_node_left, maxDepth, final_depth)
    d2 = induce_tree_maxDepth(right_x, right_y, classes, node_level+1, child_node_right, maxDepth, final_depth)

    if d1 > final_depth:
        final_depth = d1
    if d2 > final_depth:
        final_depth = d2

    return final_depth


def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with three elements: 
            - a numpy array containing the train indices
            - a numpy array containing the val indices 
            - a numpy array containing the test indices
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)
    
    folds = []
    for k in range(n_folds):

        test_indices = split_indices[k]
        valInd = 0
        if k != n_folds-1:
          valInd = k+1
        val_indices = split_indices[valInd]

        train_indices = []
        for i in range(n_folds):
          if i != k and i != valInd:
            train_indices.extend(split_indices[i])

        folds.append([train_indices, val_indices, test_indices])
        
    return folds