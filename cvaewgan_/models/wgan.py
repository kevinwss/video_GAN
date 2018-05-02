"""
Wasserstain GAN:
This is an implementation of "improved" version of Wasserstein GAN.
Gulrajani et al., "Improved Training of Wasserstein GAN", arXiv preprint, 2017.
"""

import numpy as np
import tensorflow as tf

from .base import BaseModel
from .utils import *

class Generator(object):
    def __init__(self, input_shape, z_dims):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.name = 'generator'
        self.input_shape = input_shape
        self.z_dims = z_dims

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.reshape(inputs, [-1, 1, 1, self.z_dims])
                x = tf.layers.conv2d_transpose(x, 256, (w, w), (1, 1), 'valid',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True
        return x

class Discriminator(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.variables = None
        self.update_ops = None
        self.name = 'discriminator'
        self.reuse = False

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv5'):
                w = self.input_shape[0] // (2 ** 4)
                x = tf.layers.conv2d(x, 1, (w, w), (1, 1), 'valid',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.005))
                y = tf.reshape(x, [-1, 1])

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True
        return y

class WGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='wgan',
        **kwargs
    ):
        super(WGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims
        self.n_critic = 2
        self.lmbda = 10.0

        self.gen_loss = None
        self.dis_loss = None
        self.gen_train_op = None
        self.dis_train_op = None

        self.x_train = None
        self.e_random = None
        self.batch_idx = 0

        self.z_test = None
        self.x_test = None

        self.build_model()

    def train_on_batch(self, x_batch, index):
        batchsize = x_batch.shape[0]
        self.batch_idx += 1

        z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
        eps = float(np.random.uniform(0.0, 1.0, size=(1)))
        _, g_loss, d_loss = self.sess.run(
            (self.dis_train_op, self.gen_loss, self.dis_loss),
            feed_dict={
                self.x_train: x_batch,
                self.z_train: z_sample,
                self.e_random: eps
            }
        )

        if self.batch_idx % self.n_critic == 0:
            z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
            eps = float(np.random.uniform(0.0, 1.0, size=(1)))
            _, g_loss, d_loss = self.sess.run(
                 (self.gen_train_op, self.gen_loss, self.dis_loss),
                feed_dict={
                    self.x_train: x_batch,
                    self.z_train: z_sample,
                    self.e_random: eps,
                    self.z_test: self.test_data
                }
            )

        # Summary update
        summary_priod = 1000
        if index // summary_priod != (index - batchsize) // summary_priod:
            z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
            eps = float(np.random.uniform(0.0, 1.0, size=(1)))
            summary = self.sess.run(
                self.summary,
                feed_dict={
                    self.x_train: x_batch,
                    self.z_train: z_sample,
                    self.e_random: eps,
                    self.z_test: self.test_data
                }
            )
            self.writer.add_summary(summary, index)

        return [
            ('g_loss', g_loss), ('d_loss', d_loss)
        ]

    def predict(self, z_samples):
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.uniform(-1, 1, size=(self.test_size * self.test_size, self.z_dims))

    def build_model(self):
        # Trainer
        self.f_dis = Discriminator(self.input_shape)
        self.f_gen = Generator(self.input_shape, self.z_dims)

        x_shape = (None,) + self.input_shape
        z_shape = (None,) + (self.z_dims,)
        self.x_train = tf.placeholder(tf.float32, shape=x_shape)
        self.z_train = tf.placeholder(tf.float32, shape=z_shape)
        self.e_random = tf.placeholder(tf.float32, shape=())

        x_fake = self.f_gen(self.z_train)
        y_fake = self.f_dis(x_fake)
        y_real = self.f_dis(self.x_train)

        gen_optim = tf.train.AdamOptimizer(learning_rate=1.0e-4, beta1=0.0, beta2=0.9)
        dis_optim = tf.train.AdamOptimizer(learning_rate=1.0e-4, beta1=0.0, beta2=0.9)

        x_hat = self.e_random * self.x_train + (1.0 - self.e_random) * x_fake
        y_hat = self.f_dis(x_hat)
        d_grad = tf.gradients(y_hat, [x_hat])
        d_reg = tf.square(1.0 - tf.sqrt(tf.reduce_sum(tf.square(d_grad))))

        self.gen_loss = -tf.reduce_mean(y_fake)
        self.dis_loss = -tf.reduce_mean(y_real) + tf.reduce_mean(y_fake) + self.lmbda * d_reg

        gen_optim_min = gen_optim.minimize(self.gen_loss, var_list=self.f_gen.variables)
        with tf.control_dependencies([gen_optim_min] + self.f_gen.update_ops):
            self.gen_train_op = tf.no_op(name='gen_train')

        dis_optim_min = dis_optim.minimize(self.dis_loss, var_list=self.f_dis.variables)

        with tf.control_dependencies([dis_optim_min] + self.f_dis.update_ops):
            self.dis_train_op = tf.no_op(name='dis_train')

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.f_gen(self.z_test, training=False)

        x_tile = self.image_tiling(self.x_test, self.test_size, self.test_size)

        tf.summary.image('x_real', image_cast(self.x_train), 10)
        tf.summary.image('x_fake', image_cast(x_fake), 10)
        tf.summary.image('x_tile', image_cast(x_tile), 1)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('dis_loss', self.dis_loss)
        self.summary = tf.summary.merge_all()
