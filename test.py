import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import cv2

from models import *
from datasets import load_data, mnist, svhn

models = {
    'vae': VAE,
    'dcgan': DCGAN,
    'improved': ImprovedGAN,
    'resnet': ResNetGAN,
    'began': BEGAN,
    'wgan': WGAN,
    'lsgan': LSGAN,
    'cvae': CVAE,
    'cvaegan': CVAEGAN,
    'cvaewgan': CVAEWGAN
}
#--------------------------------------------------

class Dataset(object):
    def __init__(self):
        self.images = None

    def __len__(self):
        return len(self.images)

    def _get_shape(self):
        return self.images.shape

    shape = property(_get_shape)

class ConditionalDataset(Dataset):
    def __init__(self):
        super(ConditionalDataset, self).__init__()
        self.attrs = None
        self.attr_names = None
        
def load_images(filename, size=-1):
    dset = ConditionalDataset()
    path = "../data/ucf_sports_actions"
    index_path = path + "/" +"index.txt"
    reshape_size = 64
    #images = np.zeros((2107,reshape_size,reshape_size,3)) #initialize
    images = []
    attrs = []
    
    with open(index_path) as f:
        content = f.readlines()
        content = [line.strip() for line in content]
        index = 0
        for img_path in content:
            image = cv2.imread(path+"/"+img_path)
            #print(image)
            #image = tf.image.resize_images(image, [reshape_size, reshape_size])
            image = image / 255.0
            image =  cv2.resize(image, (reshape_size,reshape_size))
            images.append(image)
            #print(images.shape)
            #print("images type",type(images))
            #print(image.shape)
            #print("image type",type(image))
            #images[i] = image
            #attrs 0
            attrs.append([1])
            
            index += 1
            if index%100 == 0:
                print(index)
                

    images = np.stack(images,axis = 0)
    attrs = np.stack(attrs,axis = 0)
    #np.save("ucf_64.npy",images)
    print(images.shape)
    dset.images = images
    dset.attrs = attrs
    dset.attr_names = ["N"]
    
    return dset


#----------for lstm--------------------

def get_image_z(datasets):
		
        with self.sess.as_default():
            self.saver = tf.train.Saver()
            if self.resume is not None:
                print('Resume model: %s' % self.resume)
                self.load_model(self.resume)
            else:
                print("Model not resumed")
                sys.exit() 

            images = datasets.images
            #np.save("datasets_images.npy",datasets.images)
            #np.save("datasets_attr.npy",datasets.images)
            attrs = datasets.attrs
            # image_z (2700*256) 
            image_z ,z_avg,z_log_var = self.sess.run(    
                [self.z_f,self.z_avg, self.z_log_var],
                feed_dict={
                    self.x_r: images, self.c_r:attrs
                }
            )
        
        #np.save("z_avg.npy",z_avg)
        #np.save("z_log_var.npy",z_log_var)
        #np.save("image_z.npy",image_z)  
        #print("z_avg.shape",z_avg.shape)
        #print("z_avg",z_avg)
        #print("z_log_var.shape",z_log_var.shape)
        #print("z_log_var",z_log_var)
        
        print("image_z.shape",image_z.shape)
        print("image_z",image_z)
        
        return image_z
        
        
        
def reshape(image_z):   # train data for LSTM

        count_file = open("../data/ucf_sports_actions/split.txt")
        lines = count_file.readlines()
        lines = [int(line.strip()) for line in lines]   #[101,101,...]
        train_data = []
        pad = [0]*256
        #max_length = max(lines) # for padding
        max_length = 100        # for cutting
        start = 0
        print(image_z.shape)
  
        for count in lines:
            #print(start)
            data = image_z[start:start + count,:]
            start += count
            '''
            if count <max_length:
                		
                padding = [pad] * (max_length - count)
                #padding = np.array(padding)
                
                
                data = np.concatenate((data,padding),axis = 0)
                
            if count > max_length:                        
                data = data[:max_length,:]                       #cut
            '''
            train_data.append(data)
        
        #train_data =  np.stack(train_data,axis = 0)
        
        #train_data = (22,100,256)
        
        #print("train_data.shape",train_data.shape)
        
        return train_data


def get_batch(train_data, batchsize = 5):
        EOS = [1]*256
        pad = [0]*256
        num_of_train_data = len(train_data)
        select = []
        batch_train_data = []
        max_length = 100
        
        for _ in range(batchsize):
            random_index = np.random.randint(0,num_of_train_data)  # test here
            batch_train_data.append(train_data[random_index])
        
        
        #batch_train_data = np.array(batch_train_data)   #(5,x,256)
        #batch_train_data_time_major = batch_train_data.swapaxes(0, 1)  #convert to time major (100,5,256)
        #print("batch_train_data_time_major",batch_train_data_time_major.shape)
        encoder_inputs_ = []
        decoder_inputs_ = []
        decoder_targets_ = []
        
        # encoder_inputs_
        for data in batch_train_data:
            encoder_inputs_.append([data[0]]  )
       
        # decoder_targets_    s+1+0
        for index in batch_train_data:  # cut
            if len(data) > max_length:
               data = data[:max_length] 		
            data = np.concatenate((data, [EOS]) , axis = 0)
            decoder_targets_.append(data)
        
        for index in range(len(decoder_targets_)): 		#padding		
            if len(decoder_targets_[index]) < max_length + 1:
               decoder_targets_[index] = np.concatenate( (decoder_targets_[index],[pad] * (max_length +1- len(decoder_targets_[index]))),axis = 0)
        
       	# decoder_inputs_   1+s+0
        
        for data in batch_train_data:
            data = np.concatenate(([EOS],data) , axis = 0)
            
            if len(data) > max_length + 1:  # cut
               data = data[:max_length+1] 				
            if len(data) < max_length + 1:   # padding
               data = np.concatenate( (data,[pad] * (max_length +1- len(data))),axis = 0)
            decoder_inputs_.append(data)
            #print(len(data))
        
        #encoder_inputs_ = batch_train_data[:,0:1]
        
        
        #decoder_inputs_ = np.array([(sequence) + [EOS] for sequence in batch_train_data])
        #decoder_targets_ = np.array([[EOS] + (sequence) for sequence in batch_train_data])
        encoder_inputs_ = np.array(encoder_inputs_)
        decoder_inputs_ = np.stack(decoder_inputs_,axis = 0)
        print(decoder_inputs_)
        decoder_targets_ = np.stack(decoder_targets_,axis = 0)
        
        encoder_inputs_ = encoder_inputs_.swapaxes(0, 1)
        decoder_inputs_  = decoder_inputs_.swapaxes(0, 1)
        decoder_targets_ = decoder_targets_.swapaxes(0, 1)

        print("encoder_inputs_",encoder_inputs_.shape,encoder_inputs_)
        print("decoder_inputs_",decoder_inputs_.shape,decoder_inputs_)
        print("decoder_targets_",decoder_targets_.shape,decoder_targets_)
        
        #return encoder_inputs_ ,decoder_inputs_,decoder_targets_


def trainLSTM(train_data):
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
       
        vocab_size = 10
        input_embedding_size = 20
        dim = 256
        encoder_hidden_units = 101
        decoder_hidden_units = encoder_hidden_units
    
        encoder_inputs = tf.placeholder(shape=(None, None,dim), dtype=tf.int32, name='encoder_inputs') #modify shape to input image
        decoder_targets = tf.placeholder(shape=(None, None,dim), dtype=tf.int32, name='decoder_targets')
        decoder_inputs = tf.placeholder(shape=(None, None,dim), dtype=tf.int32, name='decoder_inputs')
        
        decoder_inputs_embedded = encoder_inputs
        encoder_inputs_embedded = decoder_inputs
        # encoder

        encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded,
        dtype=tf.float32, time_major=True,
            )

        #decoder

        del encoder_outputs

        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="plain_decoder",
        )

        #optimizer
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
        decoder_prediction = tf.argmax(decoder_logits, 2)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
        )

        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer())


        #----------
        batch_size = 5
        loss_track = []

        max_batches = 3001
        batches_in_epoch = 1000

        try:
            for batch in range(max_batches):
                encoder_inputs_ , decoder_inputs_ , decoder_targets_ = get_batch(train_data)
           
                _, l = sess.run([train_op, loss], {encoder_inputs: encoder_inputs_,
                                               decoder_inputs: decoder_inputs_,
                                               decoder_targets: decoder_targets_})
                loss_track.append(l)

                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                    predict_ = sess.run(decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                    print()
        except KeyboardInterrupt:
            print('training interrupted')
            
            
            
         
#--------------------------------------------------



def main(_):
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datasize', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    '''
    # Load datasets
    if args.dataset == 'mnist':
        datasets = mnist.load_data()
    elif args.dataset == 'svhn':
        datasets = svhn.load_data()
    else:
        #datasets = load_data(args.dataset, args.datasize)
        datasets =load_images(args.dataset)  # load
        
         
    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)
        
#--------------------------------------------------------
    print("input_shape",datasets.shape[1:])
#--------------------------------------------------------

    model = models[args.model](
        batchsize=args.batchsize,
        input_shape=datasets.shape[1:],
        attr_names=None or datasets.attr_names,
        z_dims=args.zdims,
        output=args.output,
        resume=args.resume
    )

    if args.testmode:
        model.test_mode = True

    tf.set_random_seed(12345)

    # Training loop
    datasets.images = datasets.images.astype('float32') * 2.0 - 1.0
    #--------------------------
    print("datasets.images",datasets.images.shape)
    '''
    #image_z = model.get_image_z(datasets)
    image_z = np.load("image_z.npy")
    train_data = reshape(image_z)
    get_batch(train_data)
    #batch = get_batch(train_data)
    #trainLSTM(batch)
    #print("batch.shape", batch.shape)
     
if __name__ == '__main__':
    tf.app.run(main)
