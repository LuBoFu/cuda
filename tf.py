import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU list:", tf.config.list_physical_devices('GPU'))