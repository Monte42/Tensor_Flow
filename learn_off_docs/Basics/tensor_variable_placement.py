import tensorflow as tf
# this library can be used to find available devices
from tensorflow.python.client import device_lib

# NOTE Tensorflow 2.10 and new wont use gpus on native windows

# tf.debugging.set_log_device_placement(True) # Uncomment to se variables loaction
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))


# For better performance, TensorFlow will attempt to place tensors and variables on 
# the fastest device compatible with its dtype. This means most variables are placed 
# on a GPU if one is available.

# You can manually place tensors and variables on a CPU, even when a GPU is available
with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
# print(c)

# It's possible to set the location of a variable or tensor on one device and do the 
# computation on another device. This increases delay, because data will need to be 
# copied between the multiple devices.

# You might do this, however, if you had multiple GPU workers but only want one copy 
# of the variables.

with tf.device('CPU:0'):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
    # Element-wise multiply
    k = a * b

# print(k)