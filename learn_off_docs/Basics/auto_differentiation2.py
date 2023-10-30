import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# This topic was getting rather large, so I broke it into two sections

# ================================
# Gradients of non-scalar targets
# ================================
# A gradient is fundamentally an operation on a scalar.
x = tf.Variable(2.0)
with tf.GradientTape(persistent=True) as tape:
    y0 = x**2
    y1 = 1 / x

print(tape.gradient(y0,x).numpy())
print(tape.gradient(y1,x).numpy())

# If you ask for the gradient of multiple targets, the result for each source is
# the sum of the gradients of each target.
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y0 = x**2
    y1 = 1 / x

print("Sum of Gradients", tape.gradient({'y0':y0,'y1':y1}, x).numpy())

# Similarly, if the target(s) are not scalar the gradient of the sum is calculated
x = tf.Variable(2.)
with tf.GradientTape() as tape:
    y3 = x * [3., 4.] # 3 + 4 = 7.0

print(tape.gradient(y3,x).numpy())

x = tf.linspace(-10.0, 10.0, 200+1) # Generates evenly-spaced values in an interval along a given axis.

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.nn.sigmoid(x) # gives model non linearity??

dy_dx = tape.gradient(y, x)
# Drawing Graph
plt.plot(x, y, label='y')
plt.plot(x, dy_dx, label='dy/dx')
plt.legend()
plt.show()
_ = plt.xlabel('x')
