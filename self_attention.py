import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

class SelfAttention(Layer):
    def __init__(self, trainable=True):
        super(SelfAttention, self).__init__()

    def block(self, c):
        return Sequential([
            Conv2D(c, 1, activation='relu', padding='same'),
            Reshape((-1, c))
        ])

    def build(self, input_shape):
        self.q = self.block(input_shape[-1])
        self.v = self.block(input_shape[-1])
        self.k = self.block(input_shape[-1])

    def call(self, inputs, **kwargs):
        q = self.q(inputs)
        v = self.v(inputs)
        k = self.k(inputs)
        inputs_shape = inputs.shape
        att = tf.linalg.matmul(k,q,transpose_a=True)
        att = tf.nn.softmax(att)
        out = tf.linalg.matmul(v,att)
        out = Reshape(inputs_shape[1:])(out)

        return out
