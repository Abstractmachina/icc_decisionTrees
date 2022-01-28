def read_dataset(filepath): # eg. "data/train_full.txt" 
    import numpy as np
    """Reads a given dataset.

    Args:
        filepath txt: txt file; comma del; 
        last col is dependent var which is str
        other cols are independent var which are int

    Returns:
        tuple: a tuple of 3 numpy arrays: x, y, class
    """
    x = []
    y_labels = []

    for row in open(filepath):
        if row.strip() != "": 
            row = row.strip().split(",")
            x.append(list(map(int, row[:-1]))) 
            y_labels.append(row[-1])

    classes = np.unique(y_labels) 
    x = np.array(x)
    y = np.array(y_labels)

    return (x, y, classes)