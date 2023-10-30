import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Automatic Differentiation
# -------------------------
# In mathematics and computer algebra, automatic differentiation is a set of techniques to 
# evaluate the partial derivative of a function specified by a computer program.

# Backpropagation
# ---------------
# backpropagation performs a backward pass to adjust a neural network model's parameters, aiming to minimize 
# the mean squared error (the average squared difference between the estimated values and the actual value)



# ==========
#   Intro
# ==========
# Automatic differentiation is useful for implementing machine learning algorithms 
# such as backpropagation for training neural networks.

# To differentiate automatically, TensorFlow needs to remember what operations happen 
# in what order during the forward pass. Then, during the backward pass, TensorFlow traverses 
# this list of operations in reverse order to compute gradients.

# TensorFlow "records" relevant operations executed inside the context of a tf.GradientTape onto a "tape". 
# TensorFlow then uses that tape to compute the gradients of a "recorded" computation using reverse mode differentiation.

x = tf.Variable(3.0)
print(x**2)

with tf.GradientTape() as tape:
    y = x**2

# Once you've recorded some operations, use GradientTape.gradient(target, sources) to calculate the gradient of some target

dy_dx = tape.gradient(y,x)
print(dy_dx.numpy()) # 6.0

# The above example uses scalars, but tf.GradientTape works as easily on any tensor

a = tf.Variable(tf.random.normal((3,2)), name='a')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
c = [[1., 2., 3.]]

with tf.GradientTape(persistent=True) as tape:
    y = c @ a + b
    loss = tf.reduce_mean(y**2)

# To get the gradient of loss with respect to both variables, you can pass both as 
# sources to the gradient method. The Tape can accept both lists and dictionaries.

[dl_da,dl_db] = tape.gradient(loss,[a,b])

print('loss',loss)
print('y',y)
print(a.shape, a)
print(dl_da.shape, dl_da)

# Gradient calculations, by passing a dictionary

my_vars = {
    'a':a,
    'b':b
}

grad = tape.gradient(loss, my_vars)
print(grad['a']) # dl_da above
print(grad['b']) # y above



# ================================
#  Gradients in respect to models.
# ================================
# It's common to collect tf.Variables into a tf.Module or one of its 
# subclasses (layers.Layer, keras.Model) for checkpointing and exporting.

# Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model
# The SavedModel includes a serialized description of the computation defined by the model in 
# addition to the parameter values (checkpoint)

layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[1., 2., 3.]])

with tf.GradientTape() as tape:
    #  This is the Forward pass
    y = layer(x)
    loss = tf.reduce_mean(y**2)

# Calculate gradients with respect to every trainable variable
grad = tape.gradient(loss, layer.trainable_variables)


for var, g in zip(layer.trainable_variables, grad):
    print(f'{var.name}, shape: {g.shape}')



# ==================================
# Controlling What The Tape Watches
# ==================================
# The default behavior is to record all operations after accessing a trainable tf.Variable
# The tape needs to know which operations to record in the forward pass to calculate the gradients in the backwards pass
# You can set a tf.Variable's trainable property to be Flase, this will stop the tape from watching it
# tf.Tensor(s) and tf.constant(s) and not watched by default

# A trainable variable
x0 = tf.Variable(3.0, name='x0')
x00 = tf.Variable([[4.0],[6.0]], name='x00')
# Not trainable
x1 = tf.Variable(3.0, name='x1', trainable=False)
# Not a Variable: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name='x2') + 1.0
# Not a variable
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape:
    y = (x0**2) + (x00**2) + (x1**2) + (x2**2)

grad = tape.gradient(y, [x0, x00, x1, x2, x3])

for g in grad: 
    print(g) # returns a tf.Tensor / tf.Tensor / None / None / None

# You can list the variables being watched by the tape using the GradientTape.watched_variables method
print([var.name for var in tape.watched_variables()])

# tf.GradientTape provides hooks that give the user control over what is or is not watched.
# To record gradients with respect to a tf.Tensor, you need to call GradientTape.watch(x)
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())

# to disable the default behavior of watching all tf.Variables, set watch_accessed_variables=False 
# when creating the gradient tape, then use Gradient.watch to add a variable to the watch list
x0 = tf.Variable(0.0)
x1 = tf.Variable(10.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.softplus(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)