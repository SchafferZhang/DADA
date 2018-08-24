import cPickle
import numpy as np
import os

def unpickle(file):
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    # print d.keys()
    # print d['batch_label'].shape
    # print d['fine_label'].shape
    return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((-1,3,32,32)))/128.), 'y': np.array(d['fine_labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    # maybe_download_and_extract(data_dir)
    # unpickle(os.path.join(data_dir,'cifar-100-python/cifar-100-python/train'))
    if subset=='train':
        train_data = unpickle(os.path.join(data_dir,'cifar-100-python/cifar-100-python/train'))
        trainx = train_data['x']
        trainy = train_data['y']
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-100-python/cifar-100-python/test'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')
