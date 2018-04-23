import math
import numpy as np
import tensorflow as tf

from .base import CondBaseModel
from .utils import *

from utils.layers import conv2d_transpose, dis_block_2d, linear

class Encoder(object):
    def __init__(self, input_shape, z_dims, num_attrs):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.num_attrs = num_attrs
        self.name = 'encoder'

    def __call__(self, inputs, attrs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                '''
                a = tf.reshape(attrs, [-1, 1, 1, self.num_attrs])
                a = tf.tile(a, [1, self.input_shape[0], self.input_shape[1], 1])
                x = tf.concat([inputs, a], axis=-1)
                '''
                
                x = inputs
                x = tf.layers.conv2d(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                z_avg = tf.layers.dense(x, self.z_dims)
                z_log_var = tf.layers.dense(x, self.z_dims)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        print("z_avg",z_avg.shape) #(batch,256)
        print("z_log_var",z_log_var.shape) #(batch,256)
        
        return z_avg, z_log_var

class Decoder(object):
    def __init__(self, input_shape,batchsize):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.name = 'decoder'
        self.batch_size = batchsize

    def __call__(self, inputs, attrs, training=True,batch_size= 100):
        
        if training == False:
            self.batch_size = 10
        print('batchsize',batch_size)
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('fc1'):
				
                #w = self.input_shape[0] // (2 ** 3)
                #x = tf.concat([inputs, attrs], axis=-1)
                
                x = inputs
                
                
                self.z_, _, _ = linear(x, 512 * 4 * 4, 'g_f_h0_lin', with_w=True) # 4*4
                
                self.fg_h0 = tf.reshape(self.z_, [-1, 4, 4, 512])
            
                self.fg_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h0, scope='g_f_bn0'), name='g_f_relu0')
                
                '''
                x = tf.layers.dense(x, w * w * 256)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, w, w, 256])
                '''
                
            with tf.variable_scope('conv1'):
                self.fg_h1 = conv2d_transpose(self.fg_h0, 512, [batch_size, 8, 8, 256], name='g_f_h1')
                self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'),name='g_f_relu1')
                '''
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)
                '''
                
            with tf.variable_scope('conv2'):
                self.fg_h2 = conv2d_transpose(self.fg_h1, 256, [batch_size, 16, 16, 128], name='g_f_h2')
                self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
                '''
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)
                '''
            with tf.variable_scope('conv3'):
                self.fg_h3 = conv2d_transpose(self.fg_h2, 128, [batch_size, 32, 32, 64], name='g_f_h3')
                self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
                '''
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)
                '''
                
            with tf.variable_scope('conv4'):
                self.fg_h4 = conv2d_transpose(self.fg_h3, 64, [batch_size, 64, 64, 3], name='g_f_h4')
                self.fg_fg = tf.nn.tanh(self.fg_h4, name='g_f_actvcation')
                '''
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same')
                x = tf.tanh(x)
                '''
                
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True

        return self.fg_fg

class Classifier(object):
    def __init__(self, input_shape, num_attrs):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.name = 'classifier'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                f = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(f, self.num_attrs)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True

        return y, f

class Discriminator(object):
    def __init__(self, input_shape,reuse = False):
        self.variables = None
        self.reuse = reuse
        self.input_shape = input_shape
        self.name = 'discriminator'
        #self.batch_size = 100

    def __call__(self, inputs, training=True, reuse = False, batchsize = 100):
        self.batch_size = batchsize
        if training == False:
            self.batch_size = 10
            
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            initial_dim = 64
            """ CONV BLOCK 1 """
            d_h0 = dis_block_2d(inputs, 3, initial_dim, 'block1', reuse=reuse)
            """ CONV BLOCK 2 """
            d_h1 = dis_block_2d(d_h0, initial_dim, initial_dim * 2, 'block2', reuse=reuse)
            """ CONV BLOCK 3 """
            d_h2 = dis_block_2d(d_h1, initial_dim * 2, initial_dim * 4, 'block3', reuse=reuse)
            """ CONV BLOCK 4 """
            d_h3 = dis_block_2d(d_h2, initial_dim * 4, initial_dim * 8, 'block4', reuse=reuse)
            """ CONV BLOCK 5 """
            d_h4 = dis_block_2d(d_h3, initial_dim * 8, 1, 'block5', reuse=reuse, normalize=False)
            print(d_h4.shape) #(batch*2*2*1)
            """ LINEAR BLOCK """
            f = tf.reshape(d_h4, [self.batch_size, -1]) #similar to wgantest
             #(batch,1) d_h5
            print("f",f.shape)
            d_h5 = linear(tf.reshape(d_h4, [self.batch_size, 4]), 1) #batch_size =100 assign shape=2*2=4
            y = d_h5                                        # original y is useless in wgan, use d_h5 instead
            
            '''
            #print("conv4",x.shape)
            
            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])
            #print("avg",x.shape)
             
            with tf.variable_scope('fc1'):
                f = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(f, 1)
            '''
            #print("y",y.shape)
            #print("f",f.shape)
            
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True

        return y, f,d_h5


class CVAEWGAN(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='cvaegan',
        **kwargs
    ):
        super(CVAEWGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)
        self.cropsize = 64
        self.batch_size = 100
        self.z_dims = z_dims

        # Parameters for feature matching
        self.use_feature_match = False
        self.alpha = 0.7

        self.E_f_D_r = None
        self.E_f_D_p = None
        self.E_f_C_r = None
        self.E_f_C_p = None

        self.f_enc = None
        self.f_gen = None
        self.f_cls = None
        self.f_dis = None

        self.x_r = None
        self.c_r = None
        self.z_p = None
        #image_z
        self.z_f = None

        self.z_test = None
        self.x_test = None
        self.c_test = None

        self.enc_trainer = None
        self.gen_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.gen_loss = None
        self.dis_loss = None
        self.gen_acc = None
        self.dis_acc = None
        
        self.batchsize = 100
        
        self.build_model()

    def train_on_batch(self, batch, index):
        x_r, c_r = batch
        batchsize = len(x_r)
        z_p = np.random.uniform(-1, 1, size=(len(x_r), self.z_dims))

        _, _, _, gen_loss, dis_loss, gen_acc, dis_acc = self.sess.run(
            (self.gen_trainer, self.enc_trainer, self.dis_trainer, self.gen_loss, self.dis_loss, self.gen_acc, self.dis_acc),
            feed_dict={
                self.x_r: x_r, self.z_p: z_p, self.c_r: c_r,
                self.z_test: self.test_data['z_test'], self.c_test: self.test_data['c_test']
            }
        )

        summary_priod = 1000
        # run image_z
        '''
        if index // summary_priod != (index + batchsize) // summary_priod:
            #print("x_r",x_r.shape)
            #print("z_p",z_p.shape)
            print("self.test_data['z_test']",(self.test_data['z_test']).shape)
            batchsize = np.array([len(self.test_data['z_test'])])
            summary ,image_z = self.sess.run(
                [self.summary,self.z_f],
                feed_dict={
                    self.x_r: x_r, self.z_p: z_p, self.c_r: c_r,
                    self.z_test: self.test_data['z_test'], self.c_test: self.test_data['c_test'],self.batchsize : batchsize
                }
            )
            #print("image_z.shape",image_z.shape)
            #print("image_z",image_z)
            
            self.writer.add_summary(summary, index)
        '''
        #-----------------------------------
        
        
        
        return [
            ('gen_loss', gen_loss), ('dis_loss', dis_loss),
            ('gen_acc', gen_acc), ('dis_acc', dis_acc)
        ]

#------------predict by LSTM----------------------------
    '''
    def predict_images(self,image_z):
        dim = 256
        
        self.batchsize = 101
        #------convert tensor to numpy---------------
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        image_z = image_z.eval(session=sess)
        #--------------------------------------------
        print('image_z',image_z.shape)
        image_z = np.reshape(image_z,(1,dim))
        attrs = np.array([[1]])

        with self.sess.as_default():
            self.saver = tf.train.Saver()
            if self.resume is not None:
                print('Resume model: %s' % self.resume)
                self.load_model(self.resume)
            else:
                print("Model not resumed")
                sys.exit() 

            x_f = self.sess.run(       #predict
                (self.x_f),
                feed_dict={
                    self.z_f: image_z, self.c_r:attrs
                }
            )
        #save images
        for i in range(100):
            figure = np.ones((64,64), np.float32)
            figure = images[i,:,:,:]
            figure = Image.fromarray((figure * 255.0).astype(np.uint8))
            filename = str(i) + '.png'
            figure.save(filename)
    '''           
#---------------------------------------------------
     
    def predict(self, batch):
        z_samples, c_samples = batch
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples, self.c_test: c_samples}
        )
        return x_sample

    def make_test_data(self):
        c_t = np.identity(self.num_attrs)
        c_t = np.tile(c_t, (self.test_size, 1))
        z_t = np.random.normal(size=(self.test_size, self.z_dims))
        z_t = np.tile(z_t, (1, self.num_attrs))
        z_t = z_t.reshape((self.test_size * self.num_attrs, self.z_dims))
        self.test_data = {'z_test': z_t, 'c_test': c_t}

    def build_model(self):
        self.f_enc = Encoder(self.input_shape, self.z_dims, self.num_attrs)
        self.f_gen = Decoder(self.input_shape,batchsize = self.batchsize)  #add parameter self.batchsize

        n_cls_out = self.num_attrs if self.use_feature_match else self.num_attrs + 1
        self.f_cls = Classifier(self.input_shape, n_cls_out)
        self.f_dis = Discriminator(self.input_shape)

        # Trainer
        self.x_r = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        self.c_r = tf.placeholder(tf.float32, shape=(None, self.num_attrs))

        self.z_avg, self.z_log_var = self.f_enc(self.x_r, self.c_r)

        self.z_f = sample_normal(self.z_avg, self.z_log_var)# image_z
        
        self.x_f = self.f_gen(self.z_f, self.c_r)

        self.z_p = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        x_p = self.f_gen(self.z_p, self.c_r)
        '''
        c_r_pred, f_C_r = self.f_cls(self.x_r)
        c_f, f_C_f = self.f_cls(self.x_f)
        c_p, f_C_p = self.f_cls(x_p)
        '''
        y_r, f_D_r ,_ = self.f_dis(self.x_r)
        y_f, f_D_f ,_ = self.f_dis(self.x_f)
        y_p, f_D_p ,_ = self.f_dis(x_p)

        L_KL = kl_loss(self.z_avg, self.z_log_var)

        enc_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        gen_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        #cls_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        dis_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)

        if self.use_feature_match:
            # Use feature matching (it is usually unstable)
            L_GD = self.L_GD(f_D_r, f_D_p)
            L_GC = self.L_GC(f_C_r, f_C_p, self.c_r)
            L_G = self.L_G(self.x_r, self.x_f, f_D_r, f_D_f, f_C_r, f_C_f)

            with tf.name_scope('L_D'):
                L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_r), y_r) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_f), y_f) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_p), y_p)

            with tf.name_scope('L_C'):
                L_C = tf.losses.softmax_cross_entropy(self.c_r, c_r_pred)

            self.enc_trainer = enc_opt.minimize(L_G + L_KL, var_list=self.f_enc.variables)
            self.gen_trainer = gen_opt.minimize(L_G + L_GD + L_GC, var_list=self.f_gen.variables)
            self.cls_trainer = cls_opt.minimize(L_C, var_list=self.f_cls.variables)
            self.dis_trainer = dis_opt.minimize(L_D, var_list=self.f_dis.variables)

            self.gen_loss = L_G + L_GD + L_GC
            self.dis_loss = L_D

            # Predictor
            self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
            self.c_test = tf.placeholder(tf.float32, shape=(None, self.num_attrs))

            self.x_test = self.f_gen(self.z_test, self.c_test)
            x_tile = self.image_tiling(self.x_test, self.test_size, self.num_attrs)

            # Summary
            tf.summary.image('x_real', self.x_r, 10)
            tf.summary.image('self.x_fake', self.x_f, 10)
            tf.summary.image('x_tile', x_tile, 1)
            tf.summary.scalar('L_G', L_G)
            tf.summary.scalar('L_GD', L_GD)
            tf.summary.scalar('L_GC', L_GC)
            tf.summary.scalar('L_C', L_C)
            tf.summary.scalar('L_D', L_D)
            tf.summary.scalar('L_KL', L_KL)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('dis_loss', self.dis_loss)
        else:
            # Not use feature matching (it is more similar to ordinary GANs)
            #c_r_aug = tf.concat((self.c_r, tf.zeros((tf.shape(self.c_r)[0], 1))), axis=1)
           # c_other = tf.concat((tf.zeros_like(self.c_r), tf.ones((tf.shape(self.c_r)[0], 1))), axis=1)
           
            L_G = -tf.reduce_mean(y_f) + (- tf.reduce_mean(y_p))

            self.d_cost = (tf.reduce_mean(y_f) - tf.reduce_mean(y_r) ) +( tf.reduce_mean(y_p) - tf.reduce_mean(y_r) )

            tf.summary.scalar("g_cost", L_G)
            tf.summary.scalar("d_cost", self.d_cost)

            alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
            )

            dim =  self.cropsize * self.cropsize * 3

            vid = tf.reshape(self.x_r, [self.batch_size, dim])
            fake = tf.reshape(self.x_f, [self.batch_size, dim])

            differences = fake - vid
            interpolates = vid + (alpha * differences)

            _, _, d_hat = self.f_dis(
                tf.reshape(interpolates, [self.batch_size, self.cropsize, self.cropsize, 3]), reuse=True)

            gradients = tf.gradients(d_hat, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            L_D = self.d_cost + 10 * gradient_penalty
           
           
           
            '''
            with tf.name_scope('L_G'):
                L_G = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_f), y_f) + \
                      tf.losses.sigmoid_cross_entropy(tf.ones_like(y_p), y_p)
                      #tf.losses.softmax_cross_entropy(c_r_aug, c_f) + \
                      #tf.losses.softmax_cross_entropy(c_r_aug, c_p)
            '''
            
            with tf.name_scope('L_rec'):
                # L_rec =  0.5 * tf.losses.mean_squared_error(self.x_r, self.x_f)
                L_rec =  0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.x_r, self.x_f), axis=[1, 2, 3]))
            '''
            with tf.name_scope('L_D'):
                L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_r), y_r) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_f), y_f) + \
                      tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_p), y_p)

            with tf.name_scope('L_C'):
                L_C = tf.losses.softmax_cross_entropy(c_r_aug, c_r_pred) + \
                      tf.losses.softmax_cross_entropy(c_other, c_f) + \
                      tf.losses.softmax_cross_entropy(c_other, c_p)
            '''
            self.enc_trainer = enc_opt.minimize(L_rec + L_KL, var_list=self.f_enc.variables)
            self.gen_trainer = gen_opt.minimize(L_G + L_rec, var_list=self.f_gen.variables)
            #self.cls_trainer = cls_opt.minimize(L_C, var_list=self.f_cls.variables)
            self.dis_trainer = dis_opt.minimize(L_D, var_list=self.f_dis.variables)

            self.gen_loss = L_G + L_rec
            self.dis_loss = L_D

            # Predictor
            self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
            self.c_test = tf.placeholder(tf.float32, shape=(None, self.num_attrs))

            self.x_test = self.f_gen(self.z_test, self.c_test, training = False)   #predict
            
            x_tile = self.image_tiling(self.x_test, self.test_size, self.num_attrs)

            # Summary
            tf.summary.image('x_real', self.x_r, 10)
            tf.summary.image('self.x_fake', self.x_f, 10)
            tf.summary.image('x_tile', x_tile, 1)
            tf.summary.scalar('L_G', L_G)
            tf.summary.scalar('L_rec', L_rec)
            #tf.summary.scalar('L_C', L_C)
            tf.summary.scalar('L_D', L_D)
            tf.summary.scalar('L_KL', L_KL)
            tf.summary.scalar('gen_loss', self.gen_loss)
            tf.summary.scalar('dis_loss', self.dis_loss)

        # Accuracy
        self.gen_acc = 0.5 * binary_accuracy(tf.ones_like(y_f), y_f) + \
                       0.5 * binary_accuracy(tf.ones_like(y_p), y_p)

        self.dis_acc = binary_accuracy(tf.ones_like(y_r), y_r) / 3.0 + \
                       binary_accuracy(tf.zeros_like(y_f), y_f) / 3.0 + \
                       binary_accuracy(tf.zeros_like(y_p), y_p) / 3.0

        tf.summary.scalar('gen_acc', self.gen_acc)
        tf.summary.scalar('dis_acc', self.dis_acc)

        self.summary = tf.summary.merge_all()

    def L_G(self, x_r, x_f, f_D_r, f_D_f, f_C_r, f_C_f):
        with tf.name_scope('L_G'):
            loss = tf.constant(0.0, dtype=tf.float32)
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_r, x_f), axis=[1, 2, 3]))
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_D_r, f_D_f), axis=[1]))
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_C_r, f_C_f), axis=[1]))

        return loss

    def L_GD(self, f_D_r, f_D_p):
        with tf.name_scope('L_GD'):
            # Compute loss
            E_f_D_r = tf.reduce_mean(f_D_r, axis=0)
            E_f_D_p = tf.reduce_mean(f_D_p, axis=0)

            # Update features
            if self.E_f_D_r is None:
                self.E_f_D_r = tf.zeros_like(E_f_D_r)

            if self.E_f_D_p is None:
                self.E_f_D_p = tf.zeros_like(E_f_D_p)

            self.E_f_D_r = self.alpha * self.E_f_D_r + (1.0 - self.alpha) * E_f_D_r
            self.E_f_D_p = self.alpha * self.E_f_D_p + (1.0 - self.alpha) * E_f_D_p
            return 0.5 * tf.reduce_sum(tf.squared_difference(self.E_f_D_r, self.E_f_D_p))

    def L_GC(self, f_C_r, f_C_p, c):
        with tf.name_scope('L_GC'):
            image_shape = tf.shape(f_C_r)

            indices = tf.eye(self.num_attrs, dtype=tf.float32)
            indices = tf.tile(indices, (1, image_shape[0]))
            indices = tf.reshape(indices, (-1, self.num_attrs))

            classes = tf.tile(c, (self.num_attrs, 1))

            mask = tf.reduce_max(tf.multiply(indices, classes), axis=1)
            mask = tf.reshape(mask, (-1, 1))
            mask = tf.tile(mask, (1, image_shape[1]))

            denom = tf.reshape(tf.multiply(indices, classes), (self.num_attrs, image_shape[0], self.num_attrs))
            denom = tf.reduce_sum(denom, axis=[1, 2])
            denom = tf.tile(tf.reshape(denom, (-1, 1)), (1, image_shape[1]))

            f_1_sum = tf.tile(f_C_r, (self.num_attrs, 1))
            f_1_sum = tf.multiply(f_1_sum, mask)
            f_1_sum = tf.reshape(f_1_sum, (self.num_attrs, image_shape[0], image_shape[1]))
            E_f_1 = tf.divide(tf.reduce_sum(f_1_sum, axis=1), denom + 1.0e-8)

            f_2_sum = tf.tile(f_C_p, (self.num_attrs, 1))
            f_2_sum = tf.multiply(f_2_sum, mask)
            f_2_sum = tf.reshape(f_2_sum, (self.num_attrs, image_shape[0], image_shape[1]))
            E_f_2 = tf.divide(tf.reduce_sum(f_2_sum, axis=1), denom + 1.0e-8)

            # Update features
            if self.E_f_C_r is None:
                self.E_f_C_r = tf.zeros_like(E_f_1)

            if self.E_f_C_p is None:
                self.E_f_C_p = tf.zeros_like(E_f_2)

            self.E_f_C_r = self.alpha * self.E_f_C_r + (1.0 - self.alpha) * E_f_1
            self.E_f_C_p = self.alpha * self.E_f_C_p + (1.0 - self.alpha) * E_f_2

            # return 0.5 * tf.losses.mean_squared_error(self.E_f_C_r, self.E_f_C_p)
            return 0.5 * tf.reduce_sum(tf.squared_difference(self.E_f_C_r, self.E_f_C_p))
