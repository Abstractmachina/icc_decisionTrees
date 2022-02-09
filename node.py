import numpy as np

class Node():
    def __init__(self, data = None):
        self.left_node = None
        self.right_node = None
        self.feature_index = None
        self.split_value = None
        self.classification = None
        self.data = data

    def __str__(self):
        return str(self.data)

    # was tested, didn't work as well as get_proportion. Keeping this for future development
    def most_common(self):
        fdist=dict(zip(*np.unique(self.data, return_counts=True)))
        return list(fdist)[-1]

    def get_proportion(self, proportion):
        # Purpose of this function is to return the letter that has more than
        # % proportion of the data in the node, e.g. if there were 8 As and 2 Cs
        # and proportion was set to 0.75 then A would be returned. If
        # proportion was set to 0.9 then nothing would be returned.

        unique, counts = np.unique(self.data, return_counts=True)
        counts_array = np.asarray((unique, counts)).T
        total = len(self.data)
        for entry in counts_array:
            if (int(entry[1])/total >= proportion):
                return entry[0]
                
        return None

        
