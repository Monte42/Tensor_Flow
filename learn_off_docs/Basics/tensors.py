import tensorflow as tf
import numpy as np


#  =======
#  BASICS
#  =======
# TensorFlow operates on multidimensional arrays or tensors 
# represented as tf.Tensor objects

# All tensors are immutable like Python numbers and strings: 
# you can never update the contents of a tensor, only create 
# a new one.

# These Dimensions are reffered to be "rank" or shape

# RANK: 0    SHAPE[]: Scalar   AXES: 0
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)

# RANK: 1    SHAPE[3]: Vector   AXES: 1
# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0, 12.0, 7.0, 14.0, 33.8, 45.9, 42.0, 55.7, 53.0])

# RANK: 2    SHAPE[3,2]: Matrix    AXES: 2
# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([
                                [1, 2],
                                [3, 4],
                                [5, 6]
                            ])

# Ever time we add a new "layer" (nest) of arrays to our tensor we 
# are adding an axes to the tensor 

# There can be an arbitrary number of
# axes (sometimes called "dimensions")

# RANK: 3    SHAPE[3, 2, 5]: Matrix    AXES: 3
rank_3_tensor = tf.constant([
                                [
                                    [0.0, 0.1, 0.2, 0.3, 0.4],
                                    [0.5, 0.6, 0.7, 0.8, 0.9]
                                ],
                                [
                                    [1.0, 1.1, 1.2, 1.3, 1.4],
                                    [1.5, 1.6, 1.7, 1.8, 1.9]
                                ],
                                [
                                    [2.0, 2.1, 2.2, 2.3, 2.4],
                                    [2.5, 2.6, 2.7, 2.8, 2.9]
                                ]
                            ])

# You can convert a tensor to a NumPy array either using 
# np.array or the tensor.numpy method:

nPy_arr1 = np.array(rank_2_tensor)
nPy_arr1 = rank_2_tensor.numpy()

# Tensors often contain floats and ints, but have many other 
# types, including: complex numbers & strings

# You can do basic math on tensors, including addition, 
# element-wise multiplication, and matrix multiplication

a = tf.constant([[1, 2],[3, 4]])
b = tf.constant([[1, 1],[1, 1]]) # Could have also said `tf.ones([2,2], dtype=tf.int32)`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication

# Tensors are used in all kinds of operations (or "Ops").

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# # Find the index of the largest value
print(tf.math.argmax(c))
# # Compute the softmax
print(tf.nn.softmax(c))



# ==============
# ABOUT SHAPES
# ==============
# Tensors have shapes. Some vocabulary:
# Shape: The length (number of elements) of each of the axes of a tensor.
# Rank: Number of tensor axes. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
# Axis or Dimension: A particular dimension of a tensor.
# Size: The total number of items in the tensor, the product of the shape vector's elements.

rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# But note that the Tensor.ndim and Tensor.shape attributes don't return Tensor objects. If you 
# need a Tensor use the tf.rank or tf.shape function. This difference is subtle, but it can be 
# important when building graphs (later).

print(f'Number of axes as tensor obj: {tf.rank(rank_4_tensor)}')
print(f'Shape of tensor as tensor obj: {tf.shape(rank_4_tensor)}')

# =========
# INDEXING
# =========
#  Normal Python Indexing rules apply
#  Indexing starts at 0
#  - indices will start on the back of the list and count backwards
#  [start:stop:step]

# SCALAR
# ------ 
# Indexing with a scalar removes the axis
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# Multi-axis 
# ----------
# All the same rules apply
# To navigate the multiple axis, just indices for each axis
print(rank_3_tensor[1,1,3]) # 2nd axis, 2nd row, 4 element
print(rank_3_tensor[2,0,:]) # 3rd axis, 1st row, all elements
print(rank_3_tensor[0,0,1:4]) # 1st axis, 1st row, 2nd-4th elements
print(rank_3_tensor[2,1,::2]) # 3rd axis, 2nd row, every other element
print(rank_3_tensor[2,:,2]) # 3rd axis, 2nd element of both rows

