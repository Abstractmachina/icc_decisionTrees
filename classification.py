#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import node
import numpy as np
import decision_tree_helpers
from numpy.random import default_rng


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    """

    def __init__(self):
        self.model = {}
        self.is_trained = False

    def fit(self, x, y):
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
 
        classes = np.unique(y)
        self.model = node.Node()
        decision_tree_helpers.induce_tree(x, y, classes, 0, self.model)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def predict(self, x):
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
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        for row_number in range(0, len(x)):
            self.check_nodes(x, self.model, predictions, row_number)
                
        return predictions
        
    def check_nodes(self, x, node, predictions, row_number):
        while True:
            # base case - if this node has a classification then use it and end here
            if (node.classification):
                predictions[row_number] = node.classification
                return

            # otherwise we need to check the value of x at our given row_number and feature index
            feature_value = x[row_number, node.feature_index]

            if (feature_value >= node.split_value):
                node = node.left_node
            else:
                node = node.right_node