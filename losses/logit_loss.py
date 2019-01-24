import tensorflow as tf
import tensorflow.contrib.slim as slim

import math


W_INIT = tf.contrib.layers.xavier_initializer(uniform=False)


def get_logits(embds, labels, config, w_init=W_INIT, reuse=False, scope='logits'):
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable(name='classify_weight', shape=[embds.get_shape().as_list()[-1], config['class_num']], dtype=tf.float32, initializer=w_init, regularizer=slim.l2_regularizer(config['weight_decay']), trainable=True)
        if config['loss_type'] == 'arcface':
            return calculate_arcface_logits(embds, weights, labels, config['class_num'], config['logits_scale'], config['logits_margin'])
        elif config['loss_type'] == 'softmax':
            return slim.fully_connected(embds, num_outputs=config['class_num'], activation_fn=None, normalizer_fn=None, weights_initializer=w_init, weights_regularizer=slim.l2_regularizer(config['weight_decay']))
        else:
            raise ValueError('Invalid loss type.')


def calculate_arcface_logits(embds, weights, labels, class_num, s, m):
    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    mm = sin_m * m

    threshold = math.cos(math.pi - m)

    cos_t = tf.matmul(embds, weights, name='cos_t')

    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
    return output

