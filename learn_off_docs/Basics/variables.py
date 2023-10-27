import tensorflow as tf

# A tf variable is the recommended way to represent 
# shared, persistent state your program manipulates

# A tf.Variable represents a tensor whose value 
# can be changed by running ops on it

tf.debugging.set_log_device_placement(True) # Uncomment to se variables loaction

this_tensor = tf.constant([[1,2,3],[9,8,7]])
my_first_variable = tf.Variable(this_tensor)

# Variables can be all kinds of types, just like tensors
bool_var = tf.Variable([True, False, False, True])
comp_var = tf.Variable([5 + 4j, 6 + 1j])

print(my_first_variable)
print(bool_var)
print(comp_var)