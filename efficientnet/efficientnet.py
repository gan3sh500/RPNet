import tensorflow as tf
from .efficientnet_builder import build_model_base


class EfficientNet:
    def __init__(self, data, name='efficientnet-b0'):
        self.name = name
        inputs = data['data']
        self.build(inputs)

    def global_avg(self, x):
        return tf.reduce_mean(tf.reduce_mean(x, axis=1), axis=1)

    def build(self, inputs):
        self.layers = {}
        base = 'cls{}_fc_pose_{}'
        for c in range(1, 4):
            for p in ['xyz', 'wpqr']:
                tmp = self.global_avg(build_model_base(inputs, self.name, True)[0])
                self.layers[base.format(c,p)] = tf.layers.dense(tmp, len(p))

    def load(self, sess, ignore=''):
        var_list = [x for x in tf.global_variables() if 'efficientnet' in x.name]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, tf.train.latest_checkpoint('efficient-b0/'))

