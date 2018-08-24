import numpy as np
import os
def load(data_dir,subset):
    if subset == 'train':
        array = np.load(os.path.join(data_dir,'DDSM/ddsm-train.npz'))
    elif subset == 'test':
        array = np.load(os.path.join(data_dir, 'DDSM/ddsm-test.npz'))
    else:
        NotImplementedError('There not exists such subset!!!')
    data = array['data']
    label = array['label']
    # data = np.transpose(data,(0,3,1,2))
    data = (-127.5+data) / 128.0
    ind = np.random.permutation(data.shape[0])
    data = data[ind].astype(np.float32)
    label = label[ind].astype(np.uint8)
    return data, label






