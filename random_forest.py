import numpy as np
import collections
from classification import DecisionTreeClassifier
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split
import decision_tree_helpers
import math
import node

class RandomForestClassifier(object):
    """ Random forest classifier
    
    Attributes:
    models (list): Stores all trained models that are part of our random forest
    is_trained (bool): Keeps track of whether the classifier has been trained
    total_trees (int): Parameter that denotes the # of trees we want in our forest
    p_value (int): The number of randomly sampled features to use for splitting
    data_prop (float): the best proportion of data to use for our random_forest training data
    has_pruned (bool): indicates when we have pruned a tree. If false, no need to prune next set of leaf nodes
    
    Methods:
    improved_fit(x, y): Constructs a decision tree from data X and label y
    improved_predict(x, model): Returns an np array of predictions, to be used to calculate modal
                                class label
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self, total_trees, p_value, data_prop):
        self.models = []
        self.is_trained = False
        self.total_trees = total_trees
        self.p_value = p_value
        self.data_prop = data_prop
        self.has_pruned = False
    
    def run_forest(self, x_train, y_train, x_test):
        seed = 5000
        #60012 was old one
        rg = default_rng(seed)

        #lists to store outputs
        cross_validation_acc = []
        cross_validation_std = []
        predictions_list = []

        for i, (train_indices, bootstrap_indices) in enumerate(train_test_k_fold(self.total_trees, len(x_train), self.data_prop, rg)):
            x_forest = x_train[bootstrap_indices]
            y_forest = y_train[bootstrap_indices]
            
            #train tree using random forest classifier
            self.improved_fit(x_forest, y_forest)


    def improved_fit(self, x, y):
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
  
        classes = np.unique(y)
        model = node.Node()
        decision_tree_helpers.random_forest_classifier(x, y, classes, 0, model, self.p_value)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        self.models.append(model)

    def improved_predict(self, x, model):
        # make sure model has been trained
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # store predictions in a np array
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        for row_number in range(0, len(x)):
            self.check_nodes(x, model, predictions, row_number)
        
        return predictions
        
    def check_nodes(self, x, node, predictions, row_number):
        while True:
            # base case - if this node has a classification then use it and end here
            if (node.classification):
                predictions[row_number] = node.classification
                return

            #otherwise we need to check the value of x at our given row_number and feature index
            feature_value = x[row_number, node.feature_index]

            if (feature_value >= node.split_value):
                node = node.left_node
            else:
                node = node.right_node   
    
    #prune node function
    def prune_nodes(self, x_validate, y_validate, model):
        while True:
            # reset the model to say that on this pass, the model has not yet been pruned
            model.has_pruned = False
            # prune the entire lower level of the tree. Topiary.
            self.prune_nodes_helper(x_validate, y_validate, model, model)
            # if the model has not been pruned at all then we can now exit
            if not model.has_pruned:
                break



    # takes the root node as its argument when checking the entire tree
    def prune_nodes_helper(self, x_validate, y_validate, node, model):
        initial_prediction = self.improved_predict(x_validate, model)
        initial_accuracy = compute_accuracy(y_validate, initial_prediction)

        # if we reach a node that has only terminating nodes below it...
        if (node.left_node.classification and node.right_node.classification):
            # significant letter is defined as one with >= proportion of the total
            significant_letter = node.get_proportion(0.75)
            # if we have a significant letter in the node then turn this node
            # into a terminating node
            if (significant_letter):
                # set the node to be a temporary terminating node
                node.classification = significant_letter
                # calculate accuracy afterwards
                post_prediction = self.improved_predict(x_validate, model)
                post_accuracy = compute_accuracy(y_validate, post_prediction)
                # if there's an improvement, terminate this node. left + right sent for garbage disposal
                if post_accuracy > initial_accuracy:
                    node.left_node = None
                    node.right_node = None
                    # flag to indicate that model has been pruned at least once
                    model.has_pruned = True
                # otherwise reset the node
                else:
                    node.classification = None
                # exit the recursive call, there's nothing left to do here
                return

        if (not node.left_node.classification):
            self.prune_nodes_helper(x_validate, y_validate, node.left_node, model)
        if (not node.right_node.classification):
            self.prune_nodes_helper(x_validate, y_validate, node.right_node, model)
    