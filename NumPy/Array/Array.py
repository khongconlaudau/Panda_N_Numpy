import numpy as np

# *********** Have to be nxn dimensions ***********

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"

print(a[0], a[1], a[2])  # print 1, 2, 3

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"

print("\n\n\n -------------------------------------------")

# -------------------------------------------

c = np.zeros((2,2)) # Create an array of all zeros
print(c)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

d = np.ones((1,2))  # Create an array of all ones
print(d)              # Prints "[[ 1.  1.]]"

e = np.full((2,2), 7)# Create a constant array
print(e)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

# *****
f = np.eye(2)# Create a 2x2 identity matrix
print(f)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

g = np.random.random((2,2))  # Create an array filled with random values
print(g)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
print("\n\n\n -------------------------------------------")


# -------------------Array Indexing-------------------
a1 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])


# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b1 = a1[:2, 1:3]
print(b1)

b1[0,0] = 77 # b[0, 0] is the same piece of data as a[0, 1]
print(a1[0, 1])   # Prints "77"

print("\n\n\n -------------------------------------------")

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a3 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a3[1,:] # Rank 1 view of the second row of a
row_r2 = a3[1:2,:] # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

print()
# We can make the same distinction when accessing columns of an array:
col_r1 = a3[:,1]
col_r2 = a3[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)


darr = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
print(darr[0])

print("\n\n-------------------------------")

# Integer array indexing: When you index into numpy arrays using slicing,
# the resulting array view will always be a subarray of the original array.
# In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array

# *** Using slice changes in subarray -> changes in original array
# *** In contrast,  Using integer array does not change in the original array

b3 = np.array([[1,2], [3,4], [5,6]])
# An example of integer array indexing.
# The returned array will have shape (3,) and
print(b3[[0,1,2], [0,1,0]])    # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([b3[0,0], b3[1,1], b3[2,1]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(b3[[0,0],[1,1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([b3[0,1], b3[0,1]]))

print("\n]\n\n----------------------")

# useful Trick
b4 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

# Create an array of indices
indices = np.array([0,2,0,1])

# Select one element from each row of a using the indices in b
# ******* np.arange(n) and b form direct pairs of row indices and column indices. The size of both arrays must match. This guarantees that:
# Each element of np.arange(n) is paired with one and only one element in b

print(b4[np.arange(4), indices])# Prints "[ 1  6  7 11]"


print("\n\n\n\n")

d2 = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (d2>2)# Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.
print(bool_idx)

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(d2[bool_idx])