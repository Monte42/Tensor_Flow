import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# This topic was getting rather large, so I broke it into two sections

# ================================
# Gradients of non-scalar targets   -   https://www.tensorflow.org/guide/autodiff#gradients_of_non-scalar_targets
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
# Control Flow   -   https://www.tensorflow.org/guide/autodiff#control_flow
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



# ====================================
# Cases Where A Gradient Returns None   -   https://www.tensorflow.org/guide/autodiff#cases_where_gradient_returns_none
# ====================================
# Here are 5 examples of what may cause a gradient to return None

# Target And Source Not Connected
# -------------------------------
# When a target is not connected to the source, here z is tied with y, not x
x = tf.Variable(2.0)
y = tf.Variable(3.)

with tf.GradientTape() as tape:
    z = y**2
print("Not connected ",tape.gradient(z,x))

# Replace a Variable With a Tensor
# ---------------------------------
# One common error is to inadvertently replace a tf.Variable with a tf.Tensor, 
for epoch in range(2):
    with tf.GradientTape() as tape:
        y = x+1
    print(f"Epoch: {epoch} - ",type(x).__name__, ":", tape.gradient(y, x))
    x = x + 1 # this converts x from a variable to a tensor
            # Should have used x.assign_add(1)

# Did Calculations Outside of TensorFlow
# ---------------------------------------
# The tape can't record the gradient path if the calculation exits TensorFlow
x = tf.Variable([[1.,2.],[3.,4.]], dtype=tf.float32)

with tf.GradientTape() as tape:
    x2 = x**2
    # This calculation is made with NumPy
    y = np.mean(x2, axis=0) # returns Numpy array
    y = tf.reduce_mean(y, axis=0) # Converts to Tensor / Not watched
print("Calculations outside TensorFlow ",tape.gradient(y,x))

# Took gradients through an integer or string. 
# ---------------------------------------------
# Integers and strings are not differentiable. If a calculation 
# path uses these data types there will be no gradient.
x = tf.constant(10) # *** Constant Not a Variable

with tf.GradientTape() as tape:
    tape.watch(x) # *** must tell tape to watch x because its a constant
    y = x * x
print("Int or str ",tape.gradient(y,x), "^^^^^ Thats this Warning ^^^^^") # target(y) can be tied to source(x)"constant" thanks to tape.watch

# Took Gradients Through A Stateful Object
# ------------------------------------------
# When you use the variable, the state is read. It's normal to calculate a gradient with respect 
# to a variable, but the variable's state blocks gradient calculations from going farther back
# the tape can only observe the current state, not the history that lead to it.
x0 = tf.Variable(3.0)
x1 = tf.Variable(0.0)

with tf.GradientTape() as tape:
    x1.assign_add(x0) # This way you are addind x0 through the variable state
    # x1 = x1 + x0 # This way you do the operation on the tape to be recorded
    y = x1**2

print(tape.gradient(y,x0))



# ======================
# Zeros instead of None   -   https://www.tensorflow.org/guide/autodiff#zeros_instead_of_none
# ======================
# In some cases it would be convenient to get 0 instead of None for unconnected gradients. You can 
# decide what to return when you have unconnected gradients using the unconnected_gradients argument
x = tf.Variable([2., 2.])
y = tf.Variable(3.)

with tf.GradientTape() as tape:
    z = y**2
# Returns a Tensor the same shape as x, with 0's for values.
print(tape.gradient(z, x, unconnected_gradients=tf.UnconnectedGradients.ZERO))
