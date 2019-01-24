from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from backbones import utils

resnet_arg_scope = utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.leaky_relu, scope='preact')
        if depth == depth_in:
            shortcut = utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def block(inputs, depth, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'block_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.leaky_relu, scope='preact')
        if depth == depth_in:
            shortcut = utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = utils.conv2d_same(preact, depth, 3, stride, rate=rate, scope='conv1')
        residual = slim.conv2d(residual, depth, [3, 3], stride=1, normalizer_fn=None, activation_fn=None, scope='conv2')
        # residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2_m(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              return_raw=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, utils.stack_blocks_dense], outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                        net = utils.conv2d_same(net, 64, 3, stride=1, scope='conv1')
                    # net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = utils.stack_blocks_dense(net, blocks, output_stride)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if return_raw:
                    return net, end_points
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                end_points[sc.name + '/postnorm'] = net

                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net

                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points
resnet_v2_m.default_image_size = 224


def resnet_v2_bottleneck(scope, base_depth, num_units, stride):
    return utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }] + (num_units - 1) * [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }])
resnet_v2_m.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):
    return utils.Block(scope, block, [{
        'depth': base_depth * 4,
        'stride': stride
    }] + (num_units - 1) * [{
        'depth': base_depth * 4,
        'stride': 1
    }])
resnet_v2_m.default_image_size = 224


def resnet_v2_m_50(inputs,
                 num_classes=None,
                 is_training=True,
                 return_raw=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=16, num_units=3, stride=2),
        resnet_v2_block('block2', base_depth=32, num_units=4, stride=2),
        resnet_v2_block('block3', base_depth=64, num_units=14, stride=2),
        resnet_v2_block('block4', base_depth=128, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_m_50.default_image_size = resnet_v2_m.default_image_size


def resnet_v2_m_101(inputs,
                  num_classes=None,
                  is_training=True,
                  return_raw=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_bottleneck('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_bottleneck('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2_bottleneck('block3', base_depth=256, num_units=23, stride=2),
        resnet_v2_bottleneck('block4', base_depth=512, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_m_101.default_image_size = resnet_v2_m.default_image_size


def resnet_v2_m_152(inputs,
                  num_classes=None,
                  is_training=True,
                  return_raw=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_bottleneck('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_bottleneck('block2', base_depth=128, num_units=8, stride=2),
        resnet_v2_bottleneck('block3', base_depth=256, num_units=36, stride=2),
        resnet_v2_bottleneck('block4', base_depth=512, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_m_152.default_image_size = resnet_v2_m.default_image_size


def resnet_v2_m_200(inputs,
                  num_classes=None,
                  is_training=True,
                  return_raw=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_bottleneck('block1', base_depth=64, num_units=3, stride=2),
        resnet_v2_bottleneck('block2', base_depth=128, num_units=24, stride=2),
        resnet_v2_bottleneck('block3', base_depth=256, num_units=36, stride=2),
        resnet_v2_bottleneck('block4', base_depth=512, num_units=3, stride=2),
    ]
    return resnet_v2_m(inputs, blocks, num_classes, is_training=is_training, return_raw=return_raw, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_m_200.default_image_size = resnet_v2_m.default_image_size
