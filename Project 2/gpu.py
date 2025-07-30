import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Available GPU devices:")
print(tf.config.list_physical_devices('GPU'))
