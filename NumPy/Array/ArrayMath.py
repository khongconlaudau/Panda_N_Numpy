import numpy as np

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

# Sum
print(x+y)
print()
print(np.add(x,y))


# Subtract
print(x-y)
print()
print(np.subtract(x,y))

# Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication.
# We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices.
# dot is available both as a function in the numpy module and as an instance method of array objects:
v = np.array([9,10])
w = np.array([11,12])

# Inner product of vectors; both produce 219
print()
print(v.dot(w))
print(np.dot(v,w))


# Matrix / vector product; both produce the rank 1 array [29 67]
print()
print(x.dot(v))
print(np.dot(x,v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print()
print(x.dot(y))
print(np.dot(x,y))

# Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum
print()
print(np.sum(x)) # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0)) # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))


# Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays.
# The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:
print()
print(x.T) # Prints "[[1 3]
            #          [2 4]]"


x1 = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v1 = np.array([1, 0, 1])
print(x1+v1)

# Broadcasting
# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations.
# Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.
#
# For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

y1 = np.empty_like(x1)

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y1[i,:] = x1[i,:] + v1

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print()
print(y1)


# This works; however when the matrix x is very large, computing an explicit loop in Python could be slow.
# Note that adding the vector v to each row of the matrix x is equivalent to forming a matrix vv by stacking multiple copies of v vertically, then performing elementwise summation of x and vv.
# We could implement this approach like this:

vv = np.tile(v1,(4,1))
y2 = x1 + vv
print()
print(y2) # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

