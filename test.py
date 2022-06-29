import tensorflow as tf
from tensorflow.keras import backend as K
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.compat.v1.Session() as test_a:
  example = tf.random.normal([4, 4, 2, 2], mean=1, stddev=4, seed = 1)
  # Print absolute maximum value (shape: ())
  abs_max = K.max(example, axis=(0,1,2,3))
  print(abs_max.value)

  # Print maximum values along axis 0, 2 (shape: (4, 2))
  max_axis_0_2 = K.max(example, axis=(0, 2,))
  print(max_axis_0_2)

  # Print maximum values along axis 1 (shape: (4, 2, 2))
  max_axis_1 = K.max(example, axis=(1,))
  print(max_axis_1.shape)

  # Print maximum values along axis 0, 2, and keep reduced axes
  # reduced axes will be size 1
  max_axis_0_2_keep = K.max(example, axis=(0, 2,), keepdims=True)
  print(max_axis_0_2_keep.shape)