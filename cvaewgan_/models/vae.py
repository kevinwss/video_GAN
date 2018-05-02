import tensorflow as tf
import numpy as np

from .base import BaseModel
from .utils import *
from .wnorm import *

class Encoder(object):
    def __init__(self, input_shape, z_dims, use_wnorm=True):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.use_wnorm = use_wnorm

    def __call__(self, inputs, training=True):
        with tf.variable_scope('encoder', reuse=self.reuse):
            with tf.variable_scope('conv1'):
                if self.use_wnorm:
                    x = conv2d_wnorm(inputs, 64, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 128, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 256, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 512, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 4)
                if self.use_wnorm:
                    z_avg = conv2d_wnorm(x, self.z_dims, (w, w), (1, 1), 'valid', use_scale=True)
                    z_log_var = conv2d_wnorm(x, self.z_dims, (w, w), (1, 1), 'valid', use_scale=True)
                else:
                    z_avg = tf.layers.conv2d(x, self.z_dims, (w, w), (1, 1), 'valid')
                    z_log_var = tf.layers.conv2d(x, self.z_dims, (w, w), (1, 1), 'valid')

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='encoder')
        self.reuse = True

        return z_avg, z_log_var

class Decoder(object):
    def __init__(self, input_shape, z_dims, use_wnorm=True):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.use_wnorm = use_wnorm

    def __call__(self, inputs, training=True):
        with tf.variable_scope('decoder', reuse=self.reuse):
            with tf.variable_scope('deconv1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.reshape(inputs, [-1, 1, 1, self.z_dims])
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 256, (w, w), (1, 1), 'valid', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 256, (w, w), (1, 1), 'valid')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv2'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 256, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv3'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 128, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv4'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 64, (5, 5), (2, 2), 'same', use_scale=True)
                    x = tf.layers.batch_normalization(x, scale=False, training=training)
                else:
                    x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv5'):
                d = self.input_shape[2]
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, d, (5, 5), (1, 1), 'same', use_scale=True)
                else:
                    x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same')
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='decoder')
        self.reuse = True

        return x

class VAE(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='vae',
        **kwargs
    ):
        super(VAE, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims
        self.use_wnorm = False

        self.encoder = None
        self.decoder = None
        self.rec_loss = None
        self.kl_loss = None
        self.train_op = None

        self.x_train = None

        self.z_test = None
        self.x_test = None

        self.build_model()

    def train_on_batch(self, x_batch, index):
        _, rec_loss, kl_loss, summary = self.sess.run(
            (self.train_op, self.rec_loss, self.kl_loss, self.summary),
            feed_dict={self.x_train: x_batch, self.z_test: self.test_data}
        )
        self.writer.add_summary(summary, index)
        return [ ('rec_loss', rec_loss), ('kl_loss', kl_loss) ]

    def predict(self, z_samples):
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.normal(size=(self.test_size * self.test_size, self.z_dims))

    def build_model(self):
        self.encoder = Encoder(self.input_shape, self.z_dims, self.use_wnorm)
        self.decoder = Decoder(self.input_shape, self.z_dims, self.use_wnorm)

        # Trainer
        batch_shape = (None,) + self.input_shape
        self.x_train = tf.placeholder(tf.float32, shape=batch_shape)

        z_avg, z_log_var = self.encoder(self.x_train)
        z_sample = sample_normal(z_avg, z_log_var)
        x_sample = self.decoder(z_sample)

        rec_loss_scale = tf.constant(np.prod(self.input_shape), tf.float32)
        self.rec_loss = tf.losses.absolute_difference(self.x_train, x_sample) * rec_loss_scale
        self.kl_loss = kl_loss(z_avg, z_log_var)

        optim = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        fmin = optim.minimize(self.rec_loss + self.kl_loss)

        with tf.control_dependencies([fmin] + self.encoder.update_ops + self.decoder.update_ops):
            self.train_op = tf.no_op(name='train')

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.decoder(self.z_test)
        x_tile = self.image_tiling(self.x_test, self.test_size, self.test_size)

        # Summary
        tf.summary.image('x_real', image_cast(self.x_train), 10)
        tf.summary.image('x_fake', image_cast(x_sample), 10)
        tf.summary.image('x_tile', image_cast(x_tile), 1)
        tf.summary.scalar('rec_loss', self.rec_loss)
        tf.summary.scalar('kl_loss', self.kl_loss)

        self.summary = tf.summary.merge_all()
