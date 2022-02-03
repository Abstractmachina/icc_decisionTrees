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

####TODO: think about how we adapt this to handle multivariate trees (i stored as list, some sort of linked list in python)
def calculate_best_info_gain(x, y, classes):
    DB_entrophy = calculate_entrophy(x, y, classes)
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
            ######### TODO: Consider the case that dataset is fed floats
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
            # DONE: update max info gained, best feature, and i if info gained is higher than current best
            if info_gained > current_max_info_gained:
                current_max_info_gained = info_gained
                current_best_feature_index = feature_index
                current_best_i = i

    #print(current_max_info_gained, current_best_feature_index, current_best_i)
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
        #print(y[0])
        return
    if len(np.unique(x, axis=0)) <= 1:
        #TODO: should do a count of most commonly occuring class, and return that in node
        unique, frequency = np.unique(y, return_counts=True) #unique array with corresponding count
        parent_node["terminating_node"] = unique[np.argmax(frequency)] # place value into terminating
        #print(unique[np.argmax(frequency)])
        return 
    
    
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
### Confusion matrix
def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row), 
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion

def precision(y_gold, y_prediction):
    """ Compute the precision score per class given the ground truth and predictions
        
    Also return the macro-averaged precision across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])  

    # Compute the macro-averaged precision
    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)
    
    return (p, macro_p)

def recall(y_gold, y_prediction):
    """ Compute the recall score per class given the ground truth and predictions
        
    Also return the macro-averaged recall across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the 
                recall for class c
            - macro-recall is macro-averaged recall (a float) 
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])    

    # Compute the macro-averaged recall
    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)
    
    return (r, macro_r)

def f1_score(y_gold, y_prediction):
    """ Compute the F1-score per class given the ground truth and predictions
        
    Also return the macro-averaged F1-score across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the 
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float) 
    """

    (precisions, macro_p) = precision(y_gold, y_prediction)
    (recalls, macro_r) = recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)
    
    return (f, macro_f)

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.
    
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds




