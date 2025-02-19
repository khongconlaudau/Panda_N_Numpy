import numpy as np


# X_train with 3 features:
# - Ear Shape (1 if pointy, 0 otherwise)
# - Face Shape (1 if round, 0 otherwise)
# - Whiskers (1 if present, 0 otherwise)
X_train = np.array([[1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])
# - 1 if the animal is a cat
# - 0 otherwise
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])




def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def split_indices(X,index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have
        that feature = 1 and the right node those that have the feature = 0
        index feature = 0 => ear shape
        index feature = 1 => face shape
        index feature = 2 => whiskers
        """
    left_indices= []
    right_indices = []

    for i, x in enumerate(X):
        left_indices.append(i) if x[index_feature] == 1 else right_indices.append(i)

    return left_indices, right_indices


print(split_indices(X_train,0))

# w_left and w_right: the proportion of each animal in each node
# p_left and p_right: the proportion of each cat in each split
def weight_entropy(X,y,left_indices, right_indices):
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)

    return w_left * entropy(p_left) + w_right * entropy(p_right)


def information_gain(X,y,left_indices,right_indices):
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)

    return h_node - weight_entropy(X,y,left_indices,right_indices)


for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train,i)
    i_gain = information_gain(X_train,y_train,left_indices,right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
