import numpy as np
from openpyxl.styles.builtins import currency_0
from tensorflow.python.distribute.device_util import current


def compute_entropy(y):
    """
        Computes the entropy for

        Args:
           y (ndarray): Numpy array indicating whether each example at a node is
               positive (`1`) or negative (`0`)

        Returns:
            entropy (float): Entropy at that node

     """
    p = np.mean(y)

    return 0 if p == 0 or p == 1 else -p*np.log2(p) - (1 - p)*np.log2(1-p)

def split_dataset(X, node_indices, feature):
    """
      Splits the data at the given node into
      left and right branches

      Args:
          X (ndarray):             Data matrix of shape(n_samples, n_features)
          node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
          feature (int):           Index of feature to split on

      Returns:
          left_indices (list):     Indices with feature value == 1
          right_indices (list):    Indices with feature value == 0
    """

    left_indices, right_indices = [], []

    for i in node_indices:
        left_indices.append(i) if X[i,feature] == 1 else right_indices.append(i)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    """
       Compute the information of splitting the node on a given feature

       Args:
           X (ndarray):            Data matrix of shape(n_samples, n_features)
           y (array like):         list or ndarray with n_samples containing the target variable
           node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
           feature (int):           Index of feature to split on

       Returns:
           cost (float):        Cost computed

    """

    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    X_node, y_node = X[node_indices], y[node_indices]
    y_left = y[left_indices] if len(left_indices) > 0 else 0
    y_right = y[right_indices] if len(right_indices) > 0 else 0

    # compute entropy
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left) if len(y_left) > 0 else 0
    right_entropy = compute_entropy(y_right) if len(y_right) > 0 else 0

    total_samples = len(y_node)
    w_left = len(y_left) / total_samples if total_samples > 0 else 0
    w_right = len(y_right) / total_samples if total_samples > 0 else 0

    weighted_entropy = (w_left * left_entropy) + (w_right * right_entropy)

    information_gain =  node_entropy - weighted_entropy

    return information_gain

def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    num_features = X.shape[1]
    information_gain_list = np.zeros(num_features)
    for feature in range(num_features):
        information_gain[feature] = compute_information_gain(X, y, node_indices, feature)

    return np.argmax(information_gain_list)


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, curent_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree.
        current_depth (int):    Current depth. Parameter used during recursive call.

    """
    tree = []
    # Based condition
    if curent_depth == max_depth:
        formatting = " " * current_depth + "-" * curent_depth
        print(formatting, "%s leaf node with indices " % branch_name, node_indices)
        return

    # get best feature at this node
    best_feature = get_best_split(X, y, node_indices)

    formatting = "-" * curent_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, curent_depth, branch_name, best_feature))

    # split data for the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    # call recursive tree
    build_tree_recursive(X, y, left_indices,"left", max_depth, curent_depth + 1)
    build_tree_recursive(X, y, right_indices,"right", max_depth, curent_depth + 1)

    return tree






