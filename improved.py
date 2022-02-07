import numpy as np
from read_data import read_dataset
import math
from numpy.random import default_rng

(x, y, classes) = read_dataset("data/simple2.txt")

# print(len(y)) # 2596
# print(x[0,:]) # [5 9 6 7 1 0 5]
# print(len(x[0,:])) # 7
# print(classes) # ['A' 'C' 'E' 'G']
# print(y)       # ['G' 'C' 'G' ... 'G' 'C' 'E'] 
# print(x) 
"""
[[ 5  9  6 ...  1  0  5]
 [ 8 12  7 ...  2  1  4]
 [ 6  9  7 ...  3  2  4]
 ...
 [ 4 10  7 ...  6  8  8]
 [ 9 13  6 ...  3  2  4]
 [ 3 10  7 ...  3  5  6]]
"""
# print("_______")
# print(x[y == 'C'])

def calculate_entrophy(x, y, classes):
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

    entrophy = 0.0
    for label, value in freq_labels.items():
        if (value != 0):
            probability = (value / current_num_obs)
            if (probability > 0 and probability < 1):
                entrophy -= probability * math.log(probability, 2)

    # sanity check:
    if (sum(freq_labels.values()) != current_num_obs):
        print("Something went wrong.")
    assert(sum(freq_labels.values()) == current_num_obs)
    
    return entrophy

def make_opposite_filter(i, feature_index, x):
    opposite_filter = (x[:, feature_index] < i)
    #opposite_filter = np.array(opposite_filter)
    return opposite_filter

def calculate_best_feature(x, y, classes, DB_entrophy):
    num_of_features = len(x[0,:])
    num_of_rows = len(x)
    best_feature_index = 0
    best_info_gain = 0

    for feature_index in range(num_of_features):
        info_gained = 0.0
        unique_values, freq = np.unique(x[:,feature_index], return_counts=True)
        #print(unique_values)
        #print(freq)

        #calculate best info gain based on categorical split
        for i, label in enumerate(unique_values):
            filtering = (x[:,feature_index] == label)
            filtered_x = x[filtering, :]
            filtered_y = y[filtering]
            info_gained += calculate_entrophy(filtered_x, filtered_y, classes) * (freq[i]/num_of_rows)
        
        if (DB_entrophy - info_gained) > best_info_gain:
            best_info_gain = DB_entrophy - info_gained
            best_feature_index = feature_index
    return best_feature_index

####TODO: think about how we adapt this to handle multivariate trees (i stored as list, some sort of linked list in python)
def calculate_best_info_gain(x, y, classes):
    DB_entrophy = calculate_entrophy(x, y, classes)
    num_of_features = len(x[0,:])
    num_of_obs = len(y)

    ### Getting optiomal splitting rule:
    # Store current max info gain, the feature index, and the splitting value
    current_max_info_gained = 0.0
    current_best_i = 0

    best_feature_index = calculate_best_feature(x, y, classes, DB_entrophy)
    #print(best_feature_index)
    # Iterate over all features :

    unique_values = np.unique(x[:,best_feature_index])
    # For each feature (column):
    for i in unique_values: 
    #for i in range(col_value_max+1):
        ######### TODO: Consider the case that dataset is fed floats
        # LEFT: 
        filtering = (x[:, best_feature_index] >= i)
        filtered_x_left = x[filtering, :]
        filtered_y_left = y[filtering]
        # calculate entrophy for this particular split (left side):
        entrophy_left = calculate_entrophy(filtered_x_left, filtered_y_left, classes) 
        # RIGHT:
        opposite_filtering = make_opposite_filter(i, best_feature_index, x)
        filtered_x_right = x[opposite_filtering, :]
        filtered_y_right = y[opposite_filtering]
        # calculate entrophy for this particular split (right side):
        entrophy_right = calculate_entrophy(filtered_x_right, filtered_y_right, classes)
        # Information gained:
        proportion = len(filtered_y_left) / (len(filtered_y_left) + len(filtered_y_right))
        info_gained = DB_entrophy - (proportion * entrophy_left + (1 - proportion) * entrophy_right)
        # DONE: update max info gained, best feature, and i if info gained is higher than current best
        if info_gained > current_max_info_gained:
            current_max_info_gained = info_gained
            current_best_i = i

    #print(current_max_info_gained, current_best_feature_index, current_best_i)
    return (best_feature_index, current_best_i)

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
    # majority of this code needs a rewrite for multi-way branching decisions but is fine
    # for our first pass at a binary decision classifier

    unique_class, freq = np.unique(y, return_counts=True)
    proportion = np.zeros((len(freq),),dtype = float)
    total_count = np.sum(freq)
    for i in range(len(freq)):
        proportion[i] = freq[i]/total_count
        if proportion[i] > 1:
            print("Something went wrong")

    #print(node_level)
    if (len(y) == 0):
        return
    # base case
    if len(np.unique(y)) == 1:
        # I would still prefer that this worked so that it fully replaced the dict with y[0] but
        # I am struggling to get that to work. This is not a disastrous workaround.
        parent_node["terminating_node"] = y[0]
        #print(y[0])
        return
    if len(np.unique(x, axis=0)) <= 1:
        #TODO: should do a count of most commonly occuring class, and return that in node
        unique, frequency = np.unique(y, return_counts=True) #unique array with corresponding count
        parent_node["terminating_node"] = unique[np.argmax(frequency)] # place value into terminating
        #print(unique[np.argmax(frequency)])
        return 
    
    #really basic pruning
    '''if np.amax(proportion) >= hyperpar:
        parent_node["terminating_node"] = unique_class[np.argmax(proportion)]
        return'''
    
    
    (feature_index, split_value) = calculate_best_info_gain(x, y, classes)
    #print(feature_index, split_value)

    # path format is feature_index;split_value, except in the case of the last value which will
    # be the biggest number in the feature_index column
    # The path will be read during prediction as "get the value from the feature_index that matches
    # is less than the split_value" where that value could be either a new path or an actual result

    # create the nodes to the left and right that we will put either a new path into, or
    # put an actual result (A, C etc. )

    child_node_left = {}
    parent_node[str(feature_index) + ',' + str(split_value)] = child_node_left

    child_node_right = {}
    parent_node[str(feature_index) + ',' + str(0)] = child_node_right

    (left_x, left_y, right_x, right_y) = split_by_best_rule(feature_index, split_value, x, y)
    #print(left_x)
    #print(left_y)

    #check logic in entropy function for when a class has 0 counts
    induce_tree(left_x, left_y, classes, node_level+1, child_node_left)
    #print(right_x)
    #print(right_y)
    induce_tree(right_x, right_y, classes, node_level+1, child_node_right)

    return "finished"

### Evaluation (basic)
def compute_accuracy(y_gold, y_prediction):
    assert len(y_gold) == len(y_prediction)  

    if len(y_gold) == 0:
        return 0

    return np.sum(y_gold == y_prediction) / len(y_gold)