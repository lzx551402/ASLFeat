import sys

import cv2
import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from .cnn_wrapper.aslfeat import ASLFeatNet

sys.path.append('..')


class FeatModel(BaseModel):
    output_tensors = ["desc:0", "kpt:0", "score:0"]
    default_config = {'max_dim': 1280}

    def _init_model(self):
        return

    def _run(self, data):
        assert len(data.shape) == 3
        max_dim = max(data.shape[0], data.shape[1])
        downsample_ratio = 1
        if max_dim > self.config['max_dim']:
            downsample_ratio = self.config['max_dim'] / float(max_dim)
            data = cv2.resize(data, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
            data = data[..., np.newaxis]
        feed_dict = {"input:0": np.expand_dims(data, 0)}
        returns = self.sess.run(self.output_tensors, feed_dict=feed_dict)
        desc = np.squeeze(returns[0], axis=0)
        kpt = np.squeeze(returns[1], axis=0)
        kpt /= downsample_ratio
        score = np.squeeze(returns[2], axis=0)
        return desc, kpt, score

    def _construct_network(self):
        ph_imgs = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1), name='input')
        mean, variance = tf.nn.moments(
            tf.cast(ph_imgs, tf.float32), axes=[1, 2], keep_dims=True)
        norm_input = tf.nn.batch_normalization(ph_imgs, mean, variance, None, None, 1e-5)
        config_dict = {'det_config': self.config['config']}
        tower = ASLFeatNet({'data': norm_input}, is_training=False, resue=False, **config_dict)
