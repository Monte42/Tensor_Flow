import tensorflow as tf

# =======
# BASICS
# =======

# A tf variable is the recommended way to represent 
# shared, persistent state your program manipulates

# A tf.Variable represents a tensor whose value 
# can be changed by running ops on it

this_tensor = tf.constant([[1,2,3],[9,8,7]])
my_first_variable = tf.Variable(this_tensor)

# Variables can be all kinds of types, just like tensors
bool_var = tf.Variable([True, False, False, True])
comp_var = tf.Variable([5 + 4j, 6 + 1j])

# Variables look & act like tensor, because they are backed my tensors
# Because of this, like tensors, the have a dtype, shape, and can be 
# exported to Numpy

print("Shape: ", my_first_variable.shape)
print("DType: ", my_first_variable.dtype)
print("As NumPy: ", my_first_variable.numpy())

# Most tensor operations will also work on variables.
print("Variable:", my_first_variable)
print("\nConvert To Tensor:", tf.convert_to_tensor(my_first_variable))
print("\nIndex of highest value:", tf.math.argmax(my_first_variable))

# NOTE: Reshaping a Tensor will be explored later, just know 
    # variables can not be reshaped (add/remove el on an axes)
# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_first_variable, [1,2,3]))

# Being as variables are backed tensors, you can reassign the tensor using 
# tf.Variable.assign

# Usually, this will reuse the backing tensors place in memory.

# If you store an existing tensor in a variable the original 
# tensor will not be changed

# This will keep the same dtype, float32
my_first_variable.assign([[4, 2, 4], [6, 3, 6]]) 

# Not allowed as it resizes the variable: 
try:
    my_first_variable.assign([1.0, 2.0, 3.0])
except Exception as e:
    print(f"{type(e).__name__}: {e}")

# When use operations you will be modifying that variables backing tensor
# If you store one variable inside an other, it will create a duplicate tensor
# Operating on one variable will not affect the other

a = tf.Variable([2.0, 3.0])
b = tf.Variable(a)
a.assign([5, 6])
# a & b have different values because they are backed by seperate 
# tensor stored in two seperate points in memory
print(a.numpy())
print(b.numpy())

# There are other assign methods to explore
print(a.assign_add([2,3]).numpy())
print(a.assign_sub([7,9]).numpy())

print('\n=========\n BREAK\n=========\n')

# ================================
# Lifecycles, naming, and watching
# ================================

# A tf.Variable instance will have the same lifecycle as other Python 
# objects.  When there are no references to a variable it is deallocated 

# Variables can also be named which can help you track and debug them. 
# You can give two variables the same name.

a = tf.Variable(this_tensor, name="Mr636")
b = tf.Variable(this_tensor + 1, name="Mr636")

# These are elementwise-unequal, despite having the same name
print(a)
print(b)
print(a == b)

# Variable names are preserved when saving and loading models. By default, 
# variables in models will be assigned variable names automatically, 
# so you don't need to assign them unless you want to.

# Although variables are important for differentiation, some variables will 
# not need to be differentiated. You can turn off gradients for a variable by 
# setting trainable to false at creation. An example of a variable that would 
# not need gradients is a training step counter.
step_counter = tf.Variable(1, trainable=False)