from improvement import train_and_predict
from read_data import read_dataset, split_dataset
from numpy.random import default_rng
from evaluate import compute_accuracy, confusion_matrix, precision, recall, f1_score

if __name__ == "__main__":
    print("Loading the training dataset...")

    (x, y, classes) = read_dataset("data/train_full.txt")

    (x_test, y_test, classes_test) = read_dataset("data/test.txt")

    # Generate a validation set
    # 0.20 reserved for validation. must take 0.4125 of remaining test set to train tree on 33% bootstrap data
    seed = 60025
    rg = default_rng(seed)
    x_train, x_validate, y_train, y_validate = split_dataset(x, y, 0.2, rg)

    acc_list = []

    hyperparameter = 50
    for i in total number of different hyperparameters:
        # run train_and predict with that hyperparameter
        # calculate accuracy with returned prediction
        # store accuracy and hyperparameter value in list
        # increment hyperparameter and run train_and_predict again
    


    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x_train, y_train, x_test, x_validate, y_validate, y_test)
    print("Predictions: {}".format(predictions))

    print("\nAccuracy of prediction: ")
    print(compute_accuracy(y_test, predictions))
