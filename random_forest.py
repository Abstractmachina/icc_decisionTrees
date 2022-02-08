import numpy as np
import json
import collections
from posixpath import split
from classification import DecisionTreeClassifier
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score, train_test_k_fold, k_fold_split
import testing
import math

class RandomForestClassifier(object):
    """ Random forest classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """
    total_trees = 0
    p_value = 0
    data_prop = 0.0

    #construct tree
    def __init__(self, total_trees=100, p_value=4, data_prop=0.67):
        self.is_trained = False
        self.total_trees = total_trees
        self.p_value = p_value
        self.data_prop = data_prop
    
    def run_forest(self, x_train, y_train, x_test):
        seed = 60025
        #60012 was old one
        rg = default_rng(seed)

        #store the models
        models = []

        #lists to store outputs
        cross_validation_acc = []
        cross_validation_std = []
        predictions_list = []

        for (train_indices, test_indices) in train_test_k_fold(self.total_trees, len(x_train), self.data_prop, rg):
            x_forest = x_train[train_indices]
            y_forest = y_train[train_indices]
            x_validate = x_train[test_indices]
            y_validate = y_train[test_indices]
            
            #train tree using random forest classifier
            new_model = self.improved_fit(x_forest, y_forest)
            models.append(new_model)

            #predict using the tree on validate set
            predictions = self.improved_predict(x_validate, new_model)
            cross_validation_acc.append(compute_accuracy(y_validate, predictions))

        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        avg_acc = sum(cross_validation_acc)/len(cross_validation_acc)
        print()
        print("\nAverage accuracy of cross validation: ")
        print(avg_acc)
        #return list of all models that have been trained
        return models


    def improved_fit(self, x, y):
        classes = np.unique(y)
            #print("Entropy for the set is: ")
            #print(testing.calculate_entrophy(x, y, classes))
        model = {}

        testing.random_forest_classifier(x, y, classes, 0, model, self.p_value)
        
        # write model to file
        with open('model.json', 'w') as f:
            f.write(json.dumps(model))

        #print("done with fitting")
        self.is_trained = True
        return model

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
        
    def check_nodes(self, x, model, predictions, row_number):
        while True:
            # loop through every key at this level of the model to see which is viable
            k = model.keys()
            #print(k)
            for key in k:
                # base case, if we reach a terminating node then set predictions[row_number] to v
                if (key == "terminating_node"):
                    predictions[row_number] = model[key]
                    return

                # split the key out into its constituent parts
                split_key = key.split(',')
                feature_index = int(split_key[0])
                value = int(split_key[1])
                #print(feature_index, value)
                #do it all again from the next node, recursively.
                if (x[row_number, feature_index] >= value):
                    model = model[key]
                    break    