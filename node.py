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