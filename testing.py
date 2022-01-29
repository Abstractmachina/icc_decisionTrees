import numpy as np
from read_data import read_dataset
import math

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
    '''opposite_filter = list()
    for i in filter:
        if i == True:
            opposite_filter.append(False) 
        else:
            opposite_filter.append(True) '''
    #opposite_filter = np.array(opposite_filter)
    return opposite_filter

####TODO: put this into a function, which returns feature index, and split value i. In main recursive, we need to assign this to a node
####TODO: think about how we adapt this to handle multivariate trees (i stored as list, some sort of linked list in python)
def calculate_best_info_gain(x, y, classes):
    DB_entrophy = calculate_entrophy(x, y, classes)
    num_of_features = len(x[0,:])
    num_of_obs = len(y)

    ### Getting optiomal splitting rule:
    # Iterate over all features :
    container = []
    for feature_index in range(num_of_features):
        # For each feature (column):
        col_value_max = x[:, feature_index].max()
        col_value_min = x[:, feature_index].min()
        ##TODO: think about optimising loop with np.unique for featureset 
        for i in range(col_value_min, col_value_max + 1): 
            ######### TODO: Consider the case that dataset is fed floats
            ######### TODO: Filtering.invert to convert trues to false etc
            # LEFT: 
            filtering = (x[:, feature_index] >= i)
            filtered_x_left = x[filtering, :]
            filtered_y_left = y[filtering]
            # calculate entrophy for this particular split (left side):
            entrophy_left = calculate_entrophy(filtered_x_left, filtered_y_left, classes) 
            # RIGHT:
            opposite_filtering = make_opposite_filter(i, feature_index, x)
            filtered_x_right = x[opposite_filtering, :]
            filtered_y_right = y[opposite_filtering]
            # calculate entrophy for this particular split (right side):
            entrophy_right = calculate_entrophy(filtered_x_right, filtered_y_right, classes)
            # Information gained:
            proportion = len(filtered_y_left) / (len(filtered_y_left) + len(filtered_y_right))
            info_gained = DB_entrophy - (proportion * entrophy_left + (1 - proportion) * entrophy_right)
            # Gathering all useful info inside the container:
            # The container is a list and its elements are nested lists.
            container.append([info_gained, feature_index, i]) # [info_gained, feature_index, i] is a list of 3 elements

    # Find max information gained and associated feature_index and i:
    ####### TODO: move this to the top, test if infogained > current info gained value, and if yes, replace feature index and i with calculated value
    current_max_info_gained = 0.0
    current_best_feature_index = 0
    current_best_i = 0
    for a_list in container:
        if current_max_info_gained < a_list[0]:         # info_gained 
            current_max_info_gained = a_list[0]         # info_gained 
            current_best_feature_index = a_list[1]      # feature_index
            current_best_i = a_list[2]                  # i

    print(current_max_info_gained, current_best_feature_index, current_best_i)
    return (current_best_feature_index, current_best_i)

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


# (a, b, c, d) = split_by_best_rule(current_best_feature_index, current_best_i, x, y)

# TESTING
#print("--------")
# print(a)
#print(b)
# print(c)
#print(d)    


### Recursion:
def induce_tree(x, y, classes, node_level, parent_node):
    # majority of this code needs a rewrite for multi-way branching decisions but is fine
    # for our first pass at a binary decision classifier
    if (len(y) == 0):
        return
    # base case
    if len(np.unique(y)) == 1:
        # I would still prefer that this worked so that it fully replaced the dict with y[0] but
        # I am struggling to get that to work. This is not a disastrous workaround.
        parent_node["terminating_node"] = y[0]
        return
    if len(np.unique(x, axis=0)) <= 1:
        #TODO: should do a count of most commonly occuring class, and returnt that in node
        return
    
    
    (feature_index, split_value) = calculate_best_info_gain(x, y, classes)

    # path format is feature_index;split_value, except in the case of the last value which will
    # be the biggest number in the feature_index column
    # The path will be read during prediction as "get the value from the feature_index that matches
    # is less than the split_value" where that value could be either a new path or an actual result

    # create the nodes to the left and right that we will put either a new path into, or
    # put an actual result (A, C etc. )

    child_node_left = {}
    parent_node[str(feature_index) + ',' + str(split_value)] = child_node_left

    child_node_right = {}
    parent_node[str(feature_index) + ',' + str(x[:, feature_index].max())] = child_node_right

    (left_x, left_y, right_x, right_y) = split_by_best_rule(feature_index, split_value, x, y)

    #check logic in entropy function for when a class has 0 counts
    induce_tree(left_x, left_y, classes, node_level+1, child_node_left)
    induce_tree(right_x, right_y, classes, node_level+1, child_node_right)

    return "finished"







