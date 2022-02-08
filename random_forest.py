import numpy as np
import collections
from posixpath import split
from classification import DecisionTreeClassifier
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split
import testing
import math
import node

class RandomForestClassifier(object):
    """ Random forest classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    #construct tree
    def __init__(self, total_trees=100, p_value=4, data_prop=0.67):
        self.models = []
        self.is_trained = False
        self.total_trees = total_trees
        self.p_value = p_value
        self.data_prop = data_prop
        self.has_pruned = False
    
    def run_forest(self, x_train, y_train, x_test):
        seed = 60025
        #60012 was old one
        rg = default_rng(seed)

        #lists to store outputs
        cross_validation_acc = []
        cross_validation_std = []
        predictions_list = []

        for i, (train_indices, test_indices) in enumerate(train_test_k_fold(self.total_trees, len(x_train), self.data_prop, rg)):
            x_forest = x_train[train_indices]
            y_forest = y_train[train_indices]
            x_validate = x_train[test_indices]
            y_validate = y_train[test_indices]
            
            #train tree using random forest classifier
            self.improved_fit(x_forest, y_forest)

            #predict using the tree on validate set
            predictions = self.improved_predict(x_validate, self.models[i])
            cross_validation_acc.append(compute_accuracy(y_validate, predictions))

        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        avg_acc = sum(cross_validation_acc)/len(cross_validation_acc)
        print()
        print("\nAverage accuracy of cross validation: ")
        print(avg_acc)


    def improved_fit(self, x, y):
       #####TODO: write if_pure(y) tests for len(np.unique(y)) == 1, then return unique[0];
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
  
        classes = np.unique(y)
        model = node.Node()
        testing.random_forest_classifier(x, y, classes, 0, model, self.p_value)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        self.models.append(model)

    def improved_predict(self, x, model):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                        class label for each instance in x
        """
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # TODO guarantee there's a better way to do this with numpy but we can look into that in future
        for row_number in range(0, len(x)):
            self.check_nodes(x, model, predictions, row_number)
        
        #print("done with predicting")
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
    