import numpy as np


# Read datasets to numpy arrays:
def read_dataset(filepath): # eg. "data/train_full.txt" 
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


# Question 1.1
    # Besides the number of instances, what is another main difference between train_full.txt 
    # and train_sub.txt?
    # Answer: the main difference is that train_sub.txt is rather unbalanced dataset 
    # in comparison to train.full.
    # See analysis below:
(x1, y1, classes1) = read_dataset("data/train_full.txt")
(x2, y2, classes2) = read_dataset("data/train_sub.txt")

print("________________________________")
print("\nFor data/train_full.txt")
print("NUMBER OF OBSERVATIONS AND FEATURES: ")
print(x1.shape) # (3900, 16)
print(y1.shape) # (3900,)
print("ALL CLASSES: ")
print(classes1) 

# Is the dataset balanced in terms of labels:
freq_class_1 = dict.fromkeys(classes1, 0) # creates a dictionary with all keys initialised to 0
# Counts the number of observations for each class:
for obs in y1:
    freq_class_1[obs] += 1
# Display:
for k, v in freq_class_1.items():
    print("Class:", k, "has", v, "number of observations.")

print("MINIMUMS: ")
print(x1.min(axis=0))
print("MAXIMUMS: ")
print(x1.max(axis=0))
print("MEANS: ")
print(x1.mean(axis=0))
print("MEDIANS: ")
print(np.median(x1, axis=0)) 
print("STANDARD DEVIATIONS: ")
print(x1.std(axis=0))

for class_label in np.unique(y1):
    print("\nClass : ", class_label)
    x1_class = x1[y1 == class_label]
    print(x1_class.min(axis=0))
    print(x1_class.max(axis=0))
    print(x1_class.mean(axis=0))    
    print(np.median(x1_class, axis=0)) 
    print(x1_class.std(axis=0)) 
    print("-----------------------")

print("________________________________")
print("\nFor data/train_sub.txt.txt: ")
print("NUMBER OF OBSERVATIONS AND FEATURES: ")
print(x2.shape) 
print(y2.shape) 
print("ALL CLASSES: ")
print(classes2) 

# Is the dataset balanced in terms of labels:
freq_class_2 = dict.fromkeys(classes2, 0) 
for obs in y2:
    freq_class_2[obs] += 1
# Display:
for k, v in freq_class_2.items():
    print("Class:", k, "has", v, "number of observations.")

print("MINIMUMS: ")
print(x2.min(axis=0))
print("MAXIMUMS: ")
print(x2.max(axis=0))
print("MEANS: ")
print(x2.mean(axis=0))
print("MEDIANS: ")
print(np.median(x2, axis=0)) 
print("STANDARD DEVIATIONS: ")
print(x2.std(axis=0))

for class_label in np.unique(y2):
    print("\nClass : ", class_label)
    x2_class = x2[y2 == class_label]
    print(x2_class.min(axis=0))
    print(x2_class.max(axis=0))
    print(x2_class.mean(axis=0))    
    print(np.median(x2_class, axis=0)) 
    print(x2_class.std(axis=0)) 
    print("-----------------------")

# Question 1.2
    # What kind of attributes are provided in the dataset 
    # (Binary? Categorical/Discrete? Integers? Real numbers?) 
    # What are the ranges for each attribute in train_full.txt?
    # Answer: all the features are Integers besides the
    # categorical variable which contains Strings (categorical/discrete values)

# What are the ranges for each attribute in train_full.txt (answer):
(num_of_obs, num_of_features) = x1.shape

for feature_index in range(num_of_features):
    minimum_value = x1[:, feature_index].min()
    maximum_value = x1[:, feature_index].max()
    print("\nRanges for the {}. attribute: ".format(feature_index))
    print("The minimum value is: {}".format(minimum_value))
    print("The maximum value is: {}".format(maximum_value))



# Question 1.3
"""
train_noisy.txt is actually a corrupted version of train_full.txt, 
where we have replaced the ground truth labels with the output of a simple 
automatic classifier. What proportion of labels in train_noisy.txt is different 
than from those in train_full.txt? (Note that the observations in both datasets are the same, 
although the ordering is different). Has the class distribution been affected? 
Specify which classes have a substantially larger or smaller number of examples 
in train_noisy.txt compared to train_full.txt.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Load the data:
(x_full, y_full, classes_full) = read_dataset("data/train_full.txt")
(x_noisy, y_noisy, classes_noisy) = read_dataset("data/train_noisy.txt")
num_of_obs_full = len(y_full)
num_of_obs_noisy = len(y_noisy)

### train_full:
# Get the frequency of each class:
freq_class_full = dict.fromkeys(classes_full, 0) 
for obs in y_full:
    freq_class_full[obs] += 1

# Display: 
print("\nFrequency for train_full:")
for key, value in freq_class_full.items():
    proportion = round(value / num_of_obs_full, 3)
    print("Class:", key, "has", value, "number of observations. Proportion:", proportion)


### train_noisy:
# Get the frequency of each class:
freq_class_noisy = dict.fromkeys(classes_noisy, 0) 
for obs in y_noisy:
    freq_class_noisy[obs] += 1

# Display: 
print("\nFrequency for train_noisy:")
for key, value in freq_class_noisy.items():
    proportion = round(value / num_of_obs_noisy, 3)
    print("Class:", key, "has", value, "number of observations. Proportion:", proportion)


# Histogram train_full:
# plt.bar(list(freq_class_full.keys()), freq_class_full.values(), label="distribution")
# plt.xlabel('Classes')
# plt.ylabel('Frequency')
# plt.title('Histogram of train_full labels')
# plt.xticks(list(freq_class_full.keys()))
# plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
# plt.show()

# # Histogram train_noisy:
# plt.bar(list(freq_class_noisy.keys()), freq_class_noisy.values(), label="distribution")
# plt.xlabel('Classes')
# plt.ylabel('Frequency')
# plt.title('Histogram of train_noisy labels')
# plt.xticks(list(freq_class_noisy.keys()))
# plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
# plt.show()

# Historgram both:
labels = freq_class_full.keys()
val1 = freq_class_full.values()
val2 = freq_class_noisy.values()

x = np.arange(len(labels))
width = 0.35 

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, val1, width, label='train_full distribution')
rects2 = ax.bar(x + width/2, val2, width, label='train_noisy distribution')

ax.set_ylabel('Classes')
ax.set_title('Histograms')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="lower right")

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.show()




