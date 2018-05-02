import h5py
import numpy as np
import cv2
import tensorflow as tf

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

class PairwiseDataset(object):
    def __init__(self, x_data, y_data):
        assert x_data.shape[1] == y_data.shape[1]
        assert x_data.shape[2] == y_data.shape[2]
        assert x_data.shape[3] == 1 or y_data.shape[3] == 1 or \
               x_data.shape[3] == y_data.shape[3]

        if x_data.shape[3] != y_data.shape[3]:
            d = max(x_data.shape[3], y_data.shape[3])
            if x_data.shape[3] != d:
                x_data = np.tile(x_data, [1, 1, 1, d])
            if y_data.shape[3] != d:
                y_Data = np.tile(y_data, [1, 1, 1, d])

        x_len = len(x_data)
        y_len = len(y_data)
        l = min(x_len, y_len)

        self.x_data = x_data[:l]
        self.y_data = y_data[:l]

    def __len__(self):
        return len(self.x_data)

    def _get_shape(self):
        return self.x_data.shape

    shape = property(_get_shape)

def load_data(filename, size=-1):
    f = h5py.File(filename)

    dset = ConditionalDataset()
    dset.images = np.asarray(f['images'], 'float32') / 255.0
    dset.attrs = np.asarray(f['attrs'], 'float32')
    dset.attr_names = np.asarray(f['attr_names'])

    if size > 0:
        dset.images = dset.images[:size]
        dset.attrs = dset.attrs[:size]

    return dset
    
   
def load_images(filename, size):
    path = "../data/ucf_sports_actions"
    index_path = path + "/" +"index.txt"
    reshape_size = 64
    images = []
    with open(index_path) as f:
        content = f.readlines()
        content = [line.strip() for line in content]
        i=0
        for img_path in content:
            image = cv2.imread(path+"/"+img_path)
            image = tf.image.resize_images(image, [reshape_size, reshape_size])
        
        #print(image.shape)
            images.append(np.array(image))
            i+=1
            if i%100==0:
                print(i)
#img = cv2.imread(path+"/"+"ucf_action/Walk-Front/001/3206-12_70000.jpg")
    images = np.array(images)
    #np.save("ucf_64.npy",images)
    print(images.shape)
    
    return images
    
    
