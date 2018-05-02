import os
import sys
import time
import math
import numpy as np
from PIL import Image

import tensorflow as tf



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

        for count in lines:
            data = image_z[start:start + count,:]
            start += count
            if count <max_length:
                #data += [pad for _ in range(max_length-count)]   #padding
                data = data[:max_length,:]                        #cut
            train_data.append(data)
        train_data = np.array(train_data)
        #train_data = (22,100,256)
        
        print("train_data.shape",train_data.shape)
        
        return train_data
