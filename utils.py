from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
import tensorflow as tf


# 注意力层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        # print(output)
        return K.sum(output, axis=1)


# 自注意力层
class SelfAttentionLayer(Layer):
    def __init__(self, scale_value=1, **kwargs):
        self.scale_value = scale_value
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], input_shape[-1]),
                                      initializer='uniform',
                                      trainable=True)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        WQ = K.dot(inputs, self.kernel[0])
        WK = K.dot(inputs, self.kernel[1])
        WV = K.dot(inputs, self.kernel[2])
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        if self.scale_value < 0:
            raise 'scale_value必须大于0'
        QK = QK / (self.scale_value ** 0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK, WV)
        # 对第二个维度进行求平均（降维）: (None, 100, 100) -> (None, 100)
        output = tf.reduce_mean(V, axis=1)
        return output

    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({'scale_value': self.scale_value})
        return config
