import numpy as np
import node
from read_data import read_dataset
import math
from numpy.random import default_rng
import random


def calculate_entrophy(y, classes):
    """Calculates entrophy 

    Args:
        x int numpy array: matrix of numpy arrays
        y String nympy array: numpy array
        classes String: labels

    Returns:
        [float]: [entrophy]
    """
    current_num_obs = len(y)
    freq_labels = dict.fromkeys(classes, 0)
    for label in y:
        freq_labels[label] += 1

    # sanity check to ensure all observations accounted for
    assert(sum(freq_labels.values()) == current_num_obs)

    entrophy = 0.0
    for label, value in freq_labels.items():
        if (value != 0):
            probability = (value / current_num_obs)
            if (probability > 0 and probability < 1):
                entrophy -= probability * math.log(probability, 2)
   
    return entrophy

def make_opposite_filter(i, feature_index, x):
    opposite_filter = (x[:, feature_index] < i)
    return opposite_filter

def calculate_best_info_gain(x, y, classes):
    # Calculate dataset entrophy pre-split
    DB_entrophy = calculate_entrophy(y, classes)
    num_of_features = len(x[0,:])
    num_of_obs = len(y)

    ### Getting optiomal splitting rule:
    # Store current max info gain, the feature index, and the splitting value
    current_max_info_gained = 0.0
    current_best_feature_index = 0
    current_best_i = 0
    # Iterate over all features :
    for feature_index in range(num_of_features):
        # Obtain array of iterable split values from values in feature index
        unique_values = np.unique(x[:,feature_index])
        # For each feature (column):
        for i in unique_values: 
            # LEFT:
            filtering = (x[:, feature_index] >= i)
            filtered_y_left = y[filtering]
            # calculate entrophy for this particular split (left side):
            entrophy_left = calculate_entrophy(filtered_y_left, classes) 
            # RIGHT:
            opposite_filtering = make_opposite_filter(i, feature_index, x)
            filtered_y_right = y[opposite_filtering]
            # calculate entrophy for this particular split (right side):
            entrophy_right = calculate_entrophy(filtered_y_right, classes)
            # Information gained:
            proportion = len(filtered_y_left) / (len(filtered_y_left) + len(filtered_y_right))
            info_gained = DB_entrophy - (proportion * entrophy_left + (1 - proportion) * entrophy_right)
            # update max info gained, best feature, and i if info gained is higher than current best
            if info_gained >= current_max_info_gained:
                current_max_info_gained = info_gained
                current_best_feature_index = feature_index
                current_best_i = i

    return (current_best_feature_index, current_best_i, current_max_info_gained)

def split_by_best_rule(current_best_feature_index, current_best_i, x, y):
    """Split the dataset so that information gained of the resulting split is maximised.

    Args:
        current_best_feature_index ([int]): [This is the column number index that maximises info gained]
        current_best_i ([int]): [This is the best integer number by which the split is done that maximises info gained]
        x and y are numpy arrays

    Returns:
        Tuple of 4 Numpy arrays X and Y for both left and right split
    """
    # LEFT:
    filtering = (x[:, current_best_feature_index] >= current_best_i)
    left_x = x[filtering, :]
    left_y = y[filtering]
    
    # RIGHT:
    opposite_filtering = make_opposite_filter(current_best_i, current_best_feature_index, x)
    right_x = x[opposite_filtering, :]
    right_y = y[opposite_filtering]

    return (left_x, left_y, right_x, right_y)

   
### Recursion:
def induce_tree(x, y, classes, node_level, parent_node):
    # Catches case if we ever pass an empty subset, which should not happen
    assert(len(y) != 0)
    
    # base case: only one class 
    if len(np.unique(y)) == 1:
        parent_node.classification = y[0]
        return True
    
    (feature_index, split_value, info_gain) = calculate_best_info_gain(x, y, classes)

    # another base case: if splitting yields no information gain, take majority label
    if info_gain == 0:
        unique, frequency = np.unique(y, return_counts=True) #unique array with corresponding count
        parent_node.classification = unique[np.argmax(frequency)] # place value into terminating
        return True

    # create the nodes to the left and right that we will put either a new path into, or
    # put an actual result (A, C etc. )
    parent_node.feature_index = feature_index
    parent_node.split_value = split_value
    parent_node.left_node = node.Node()
    parent_node.right_node = node.Node()
    parent_node.data = y

    (left_x, left_y, right_x, right_y) = split_by_best_rule(feature_index, split_value, x, y)

    induce_tree(left_x, left_y, classes, node_level+1, parent_node.left_node)
    induce_tree(right_x, right_y, classes, node_level+1, parent_node.right_node)

def random_forest_classifier(x, y, classes, node_level, parent_node, p_value):

    # Catches case if we ever pass an empty subset, which should not happen
    assert(len(y) != 0)

    # base case, if there is only 1 class left in set
    if len(np.unique(y)) == 1:
        parent_node.classification = y[0]
        return True
    
    # if subset has identical features, return most common. this has to happen before generating random features
    if len(np.unique(x, axis=0)) <= 1:
        unique, frequency = np.unique(y, return_counts=True)
        parent_node.classification = unique[np.argmax(frequency)] 
        return 

    # Generate dataset containing only features listed in random_features
    random_features = random.sample(range(0, len(x[0,:])), p_value)
    x_forest = x[:,random_features]
    
    # case where the subset with 4 features are all the same, re-randomise until this isn't the case
    # Note that any subset where every row of data is the same will have triggered a previous base case
    # and therefore there will always be a combination of 4 that can be split

    while len(np.unique(x_forest, axis=0)) == 1:
        random_features = random.sample(range(0, len(x[0,:])), p_value)
        x_forest = x[:,random_features]

    (feature_index, split_value, info_gain) = calculate_best_info_gain(x_forest, y, classes)

    # another base case: if splitting yields no information gain, take majority label
    if info_gain == 0:
        unique, frequency = np.unique(y, return_counts=True) #unique array with corresponding count
        parent_node.classification = unique[np.argmax(frequency)] # place value into terminating
        return  True

    # feature index returned is the index to the column in random_features. So convert to actual column number
    feature_index = random_features[feature_index]

    parent_node.feature_index = feature_index
    parent_node.split_value = split_value
    parent_node.left_node = node.Node()
    parent_node.right_node = node.Node()
    parent_node.data = y

    (left_x, left_y, right_x, right_y) = split_by_best_rule(feature_index, split_value, x, y)

    random_forest_classifier(left_x, left_y, classes, node_level+1, parent_node.left_node, p_value)
    random_forest_classifier(right_x, right_y, classes, node_level+1, parent_node.right_node, p_value)  
    return True  
