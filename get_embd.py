import io
import os
import yaml
import pickle
import argparse
import numpy as np
import tensorflow as tf

from scipy import misc

from model import get_embd
from eval.utils import calculate_roc, calculate_tar


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='/data/hhd/InsightFace-tensorflow/output/20190116-130753/checkpoints/ckpt-m-116000', help='model path')
    parser.add_argument('--read_path', type=str, default='', help='path to image file or directory to images')
    parser.add_argument('--save_path', type=str, default='embds.pkl', help='path to save embds')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')

    return parser.parse_args()


def load_image(path, image_size):
    print('reading %s' % path)
    if os.path.isdir(path):
        paths = list(os.listdir(path))
    else:
        paths = [path]
    images = []
    images_f = []
    for path in paths:
        img = misc.imread(path)
        img = misc.imresize(img, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img/127.5-1.0
        img_f = img_f/127.5-1.0
        images.append(img)
        images_f.append(img_f)
    fns = [os.path.basename(p) for p in paths]
    print('done!')
    return (np.array(images), np.array(images_f), fns)



def evaluate(embeddings, actual_issame, far_target=1e-3, distance_metric=0, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    if distance_metric == 1:
        thresholdes = np.arange(0, 1, 0.0025)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), distance_metric=distance_metric, nrof_folds=nrof_folds)
    tar, tar_std, far = calculate_tar(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), far_target=far_target, distance_metric=distance_metric, nrof_folds=nrof_folds)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    return tpr, fpr, acc_mean, acc_std, tar, tar_std, far


def run_embds(sess, images, batch_size, image_size, train_mode, embds_ph, image_ph, train_ph_dropout, train_ph_bn):
    if train_mode >= 1:
        train = True
    else:
        train = False
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)
        print('%d/%d' % (i, batch_num), end='\r')
    if left > 0:
        image_batch = np.zeros([batch_size, image_size, image_size, 3])
        image_batch[:left, :, :, :] = images[-left:]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)[:left]
    print()
    print('done!')
    return np.array(embds)


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'build':
        print('building...')
        config = yaml.load(open(args.config_path))
        images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
        train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
        train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
        embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
        print('done!')
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            tf.global_variables_initializer().run()
            print('loading...')
            saver = tf.train.Saver(var_list=tf.trainable_variables())
            saver.restore(sess, args.model_path)
            print('done!')

            batch_size = config['batch_size']
            imgs, imgs_f, fns = load_image(args.read_path, config['image_size'])
            print('forward running...')
            embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], args.train_mode, embds, images, train_phase_dropout, train_phase_bn)
            embds_f_arr = run_embds(sess, imgs_f, batch_size, config['image_size'], args.train_mode, embds, images, train_phase_dropout, train_phase_bn)
            embds_arr = embds_arr/np.linalg.norm(embds_arr, axis=1, keepdims=True)+embds_f_arr/np.linalg.norm(embds_f_arr, axis=1, keepdims=True)
            embds_arr = embds_arr/np.linalg.norm(embds_arr, axis=1, keepdims=True)
            print('done!')
            print('saving...')
            embds_dict = dict(*zip(fns, list(embds_arr)))
            pickle.dump(embds_dict, open(args.save_path, 'wb'))
            print('done!')

