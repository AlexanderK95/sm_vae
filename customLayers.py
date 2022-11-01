import tensorflow as tf
from tensorflow.keras import backend as K

class SampleLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(SampleLayer, self).__init__(name=name)
    
    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        sampled_point = mu + K.exp(log_variance / 2) * epsilon
        return tf.convert_to_tensor(sampled_point)