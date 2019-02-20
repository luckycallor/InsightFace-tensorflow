import os
import time
import pickle
import argparse
import numpy as np

import io
import yaml
from scipy import misc

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datetime import datetime

from losses.logit_loss import get_logits
from data.classificationDataTool import ClassificationImageData
from model import get_embd
from utils import average_gradients, check_folders, analyze_vars
from evaluate import load_bin, evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, help='path to config file', default='./configs/config_finetune.yaml')

    return parser.parse_args()


def inference(images, labels, is_training_dropout, is_training_bn, config):
    embds, end_points = get_embd(images, is_training_dropout, is_training_bn, config)
    logits = get_logits(embds, labels, config)
    end_points['logits'] = logits
    return embds, logits, end_points


class Trainer:
    def __init__(self, config):
        self.config = config
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.output_dir = os.path.join(config['output_dir'], subdir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'log')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.debug_dir = os.path.join(self.output_dir, 'debug')
        check_folders([self.output_dir, self.model_dir, self.log_dir, self.checkpoint_dir, self.debug_dir])
        self.val_log = os.path.join(self.output_dir, 'val_log.txt')

        self.batch_size = config['batch_size']
        self.gpu_num = config['gpu_num']
        if self.batch_size % self.gpu_num != 0:
            raise ValueError('batch_size must be a multiple of gpu_num')
        self.image_size = config['image_size']
        self.epoch_num = config['epoch_num']
        self.step_per_epoch = config['step_per_epoch']
        self.val_freq = config['val_freq']
        self.val_data = config['val_data']
        self.val_bn_train = config['val_bn_train']
        # for k, v in config['val_data'].items():
        #     self.val_data[k] = load_bin(v, self.image_size)
        #     imgs = self.val_data[k][0]
        #     np.save(os.path.join(self.debug_dir, k+'.npy'), imgs[:100])

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(self.config))


    def build(self):
        self.train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_dropout')
        self.train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_bn')
        self.global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        self.inc_op = tf.assign_add(self.global_step, 1, name='increment_global_step')
        scale = int(512.0/self.batch_size)
        lr_steps = [scale*s for s in self.config['lr_steps']]
        lr_values = [v/scale for v in self.config['lr_values']]
        # lr_steps = self.config['lr_steps']
        self.lr = tf.train.piecewise_constant(self.global_step, boundaries=lr_steps, values=lr_values, name='lr_schedule')

        cid = ClassificationImageData(img_size=self.image_size, augment_flag=self.config['augment_flag'], augment_margin=self.config['augment_margin'])
        train_dataset = cid.read_TFRecord(self.config['train_data']).shuffle(10000).repeat().batch(self.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        self.train_images, self.train_labels = train_iterator.get_next()
        self.train_images = tf.identity(self.train_images, 'input_images')
        self.train_labels = tf.identity(self.train_labels, 'labels')
        if self.gpu_num <= 1:
            self.embds, self.logits, self.end_points = inference(self.train_images, self.train_labels, self.train_phase_dropout, self.train_phase_bn, self.config)
            self.embds = tf.identity(self.embds, 'embeddings')
            self.inference_loss = slim.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.train_labels)
            self.wd_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_loss = self.inference_loss+self.wd_loss
            pred = tf.arg_max(tf.nn.softmax(self.logits), dimension=-1, output_type=tf.int64)
            self.train_acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.train_labels), tf.float32))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            vars_softmax = [v for v in tf.trainable_variables() if 'embd_extractor' not in v.name]
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.config['momentum']).minimize(self.train_loss)
                self.train_op_softmax = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.config['momentum']).minimize(self.train_loss, var_list=vars_softmax)
        else:
            self.embds = []
            self.logits = []
            self.inference_loss = []
            self.wd_loss = []
            self.train_loss = []
            pred = []
            tower_grads = []
            tower_grads_softmax = []
            update_ops = []
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.config['momentum'])
            train_images = tf.split(self.train_images, self.gpu_num)
            train_labels = tf.split(self.train_labels, self.gpu_num)
            for i in range(self.gpu_num):
                sub_train_images = train_images[i]
                sub_train_labels = train_labels[i]
                with tf.device('/gpu:%d' % i):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(i > 0)):
                        embds, logits, end_points = inference(sub_train_images, sub_train_labels, self.train_phase_dropout, self.train_phase_bn, self.config)
                        inference_loss = slim.losses.sparse_softmax_cross_entropy(logits=logits, labels=sub_train_labels)
                        wd_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        train_loss = inference_loss+wd_loss
                        pred.append(tf.arg_max(tf.nn.softmax(logits), dimension=-1, output_type=tf.int64))
                        vars_softmax = [v for v in tf.trainable_variables() if 'embd_extractor' not in v.name]
                        tower_grads.append(opt.compute_gradients(train_loss))
                        tower_grads_softmax.append(opt.compute_gradients(train_loss, var_list=vars_softmax))
                        update_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
                        self.embds.append(embds)
                        self.logits.append(logits)
                        self.inference_loss.append(inference_loss)
                        self.wd_loss.append(wd_loss)
                        self.train_loss.append(train_loss)
            self.embds = tf.concat(self.embds, axis=0)
            self.logits = tf.concat(self.logits, axis=0)
            self.inference_loss = tf.add_n(self.inference_loss)/self.gpu_num
            self.wd_loss = tf.add_n(self.wd_loss)/self.gpu_num
            self.train_loss = tf.add_n(self.train_loss)/self.gpu_num
            pred = tf.concat(pred, axis=0)
            self.train_acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.train_labels), tf.float32))
            train_ops = [opt.apply_gradients(average_gradients(tower_grads))]
            train_ops_softmax = [opt.apply_gradients(average_gradients(tower_grads_softmax))]
            train_ops.extend(update_ops)
            train_ops_softmax.extend(update_ops)
            self.train_op = tf.group(*train_ops)
            self.train_op_softmax = tf.group(*train_ops_softmax)


        self.train_summary = tf.summary.merge([
            tf.summary.scalar('inference_loss', self.inference_loss),
            tf.summary.scalar('wd_loss', self.wd_loss),
            tf.summary.scalar('train_loss', self.train_loss),
            tf.summary.scalar('train_acc', self.train_acc)
        ])

    def run_embds(self, sess, images):
        batch_num = len(images)//self.batch_size
        left = len(images)%self.batch_size
        embds = []
        for i in range(batch_num):
            cur_embd = sess.run(self.embds, feed_dict={self.train_images: images[i*self.batch_size: (i+1)*self.batch_size], self.train_phase_dropout: False, self.train_phase_bn: self.val_bn_train})
            embds += list(cur_embd)
        if left > 0:
            image_batch = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
            image_batch[:left, :, :, :] = images[-left:]
            cur_embd = sess.run(self.embds, feed_dict={self.train_images: image_batch, self.train_phase_dropout: False, self.train_phase_bn: self.val_bn_train})
            embds += list(cur_embd)[:left]
        return np.array(embds)

    def save_image_label(self, images, labels, step):
        save_dir = os.path.join(self.debug_dir, 'image_by_label')
        for i in range(len(labels)):
            if(labels[i] < 10):
                cur_save_dir = os.path.join(save_dir, str(labels[i]))
                check_folders(cur_save_dir)
                misc.imsave(os.path.join(cur_save_dir, '%d_%d.jpg' % (step, i)), images[i])


    def train(self):
        self.build()
        analyze_vars(tf.trainable_variables(), os.path.join(self.output_dir, 'model_vars.txt'))
        with open(os.path.join(self.output_dir, 'regularizers.txt'), 'w') as f:
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                f.write(v.name+'\n')
        # exit(-1)
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            tf.global_variables_initializer().run()
            saver_ckpt = tf.train.Saver()
            saver_best = tf.train.Saver()
            saver_embd = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if 'embd_extractor' in v.name])
            if config['pretrained_model'] != '':
                saver_embd.restore(sess, config['pretrained_model'])
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            start_time = time.time()
            best_acc = 0
            counter = 0
            debug = True
            for i in range(self.epoch_num):
                if i < config['fixed_epoch_num']:
                    cur_train_op = self.train_op_softmax
                else:
                    cur_train_op = self.train_op
                for j in range(self.step_per_epoch):
                    _, l, l_wd, l_inf, acc, s, _ = sess.run([cur_train_op, self.train_loss, self.wd_loss, self.inference_loss, self.train_acc, self.train_summary, self.inc_op], feed_dict={self.train_phase_dropout: True, self.train_phase_bn: True})
                    counter += 1
                    
                    print("Epoch: [%2d/%2d] [%6d/%6d] time: %.2f, loss: %.3f (inference: %.3f, wd: %.3f), acc: %.3f" % (i, self.epoch_num, j, self.step_per_epoch, time.time() - start_time, l, l_inf, l_wd, acc))
                    start_time = time.time()
                    if counter % self.val_freq == 0:
                        saver_ckpt.save(sess, os.path.join(self.checkpoint_dir, 'ckpt-m'), global_step=counter)
                        acc = []
                        with open(self.val_log, 'a') as f:
                            f.write('step: %d\n' % counter)
                            for k, v in self.val_data.items():
                                imgs, imgs_f, issame = load_bin(v, self.image_size)
                                embds = self.run_embds(sess, imgs)
                                embds_f = self.run_embds(sess, imgs_f)
                                embds = embds/np.linalg.norm(embds, axis=1, keepdims=True)+embds_f/np.linalg.norm(embds_f, axis=1, keepdims=True)
                                tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds, issame, far_target=1e-3, distance_metric=0)
                                f.write('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (k, acc_mean, acc_std, tar, tar_std, far))
                                acc.append(acc_mean)
                            acc = np.mean(np.array(acc))
                            if acc > best_acc:
                                saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
                                best_acc = acc

                        
if __name__ == '__main__':
    args = parse_args()
    config = yaml.load(open(args.config_path))
    trainer = Trainer(config)
    trainer.train()


                    