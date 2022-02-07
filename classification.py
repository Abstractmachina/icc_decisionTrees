#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import json
#TODO: remove posixpath?
from posixpath import split
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
        self.model = {}
        testing.induce_tree(x, y, classes, 0, self.model)
        
        # write model to file
        with open('model.json', 'w') as f:
            f.write(json.dumps(self.model))

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
        
    def check_nodes(self, x, model, predictions, row_number):
        while True:
            # loop through every key at this level of the model to see which is viable
            k = model.keys()
            #print(k)
            for key in k:
                # base case, if we reach a terminating node then set predictions[row_number] to value
                if (key == "terminating_node"):
                    predictions[row_number] = model[key]
                    return

                # split the key out into its constituent parts
                split_key = key.split(',')
                feature_index = int(split_key[0])
                value = int(split_key[1])

                #do it all again from the next node in the tree.
                if (x[row_number, feature_index] >= value):
                    model = model[key]
                    break

        
