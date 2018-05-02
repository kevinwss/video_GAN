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
    
    model.main_loop(datasets,
                    epochs=args.epoch)

if __name__ == '__main__':
    tf.app.run(main)
