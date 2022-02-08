#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import json
import node
import numpy as np
import testing


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.model = {}
        self.is_trained = False
        self._treeDepth = 1

    def fit(self, x, y, maxDepth = 0):
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
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    
        classes = np.unique(y)
        self.model = node.Node()
        testing.induce_tree(x, y, classes, 0, self.model)
        
        # write model to file
        # with open('model.json', 'w') as f:
        #     f.write(json.dumps(self.model))

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
    
    def getFinalTreeDepth(self):
        return self._treeDepth


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
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
    
        # TODO guarantee there's a better way to do this with numpy but we can look into that in future
        for row_number in range(0, len(x)):
            self.check_nodes(x, self.model, predictions, row_number)
                
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

    def prune_nodes(self):
        self.prune_nodes_helper(self.model)

    # takes the root node as its argument when checking the entire tree
    def prune_nodes_helper(self, node):
        if (node.left_node.classification and node.right_node.classification):
            significant_letter = node.get_proportion(0.75)
            # if we have a significant letter in the node then turn this node
            # into a terminating node
            if (significant_letter):
                node.classification = significant_letter
                #check accuracy and if it's better now then delete the nodes below
                # if accuracy > accuracy_before_pruning:
                    # node.left_node = None
                    # node.right_node = None
                # else:
                    # node.classification = None
                return

        if (not node.left_node.classification):
            self.prune_nodes_helper(node.left_node)
        if (not node.right_node.classification):
            self.prune_nodes_helper(node.right_node)

        
