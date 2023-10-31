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



# =============
# Control Flow
# =============
# Because a gradient tape records operations as they are executed, 
# Python control flow is naturally handled (for example, if and while statements)

# Un-comment line 56 and comment out 57 to see the difference
x = tf.constant(1.0) 
# x = tf.constant(-1.0)
v0 = tf.Variable(2.0)
v1 = tf.Variable(2.5)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    result = v0 if x > 0.0 else v1**2

dv0, dv1 = tape.gradient(result, [v0,v1])
print(dv0)
print(dv1)

# control statements themselves are not differentiable, so they are invisible to gradient-based optimizers
# Depending on the value of x in the above example, the tape either records result = v0 or result = v1**2. 
# The gradient with respect to x is always None
dx = tape.gradient(result,x)
print(dx)