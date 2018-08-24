from __future__ import print_function
import cPickle
import argparse
import time
import numpy as np
import plotting
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as ll
from lasagne.layers import dnn
from lasagne.init import Normal
from lasagne.layers import set_all_param_values
from lasagne.updates import rmsprop, sgd
import ddsm_data
import nn
from lasagne.nonlinearities import sigmoid
from inception_v3 import build_network
from alexnet import build_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--dataset', type=str, default='DDSM')
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='../datasets/')
parser.add_argument('--results_dir', type=str, default='../results/')
parser.add_argument('--aug',action='store_true')
args = parser.parse_args()

def gen_minibatches(x,y,batch_size,shuffle=False):
    assert len(x) == len(y), "Training data size don't match"
    if shuffle:
        ids = np.random.permutation(len(x))
    else:
        ids = np.arange(len(x))
    for start_idx in range(0,len(x)-batch_size+1,batch_size):
        ii = ids[start_idx:start_idx+batch_size]
        yield x[ii],y[ii]
# parameters of ImageDataGenerator
# datagen = ImageDataGenerator(featurewise_center=False,
#                              samplewise_center=False,
#                              featurewise_std_normalization=False,
#                              samplewise_std_normalization=False,
#                              zca_whitening=False,
#                              zca_epsilon=1e-6,
#                              rotation_range= 30.,
#                              width_shift_range=0.,
#                              height_shift_range=0.,
#                              shear_range=0.,
#                              zoom_range=0.,
#                              channel_shift_range=0,
#                              fill_mode='constant',
#                              cval=0.,
#                              horizontal_flip=False,
#                              vertical_flip=False,
#                              rescale=None,
#                              preprocessing_function=None,
#                              data_format=None)

aug_params = dict(rotation_range=5,zoom_range=0.02,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
if args.aug:
    datagen = ImageDataGenerator(**aug_params)
else:
    datagen = ImageDataGenerator()
# num_classes = 2
num_epoch = 600
rng = np.random.RandomState(seed=1)
trainx, trainy = ddsm_data.load(args.data_dir,subset='train')
pos = trainx[trainy==1]
neg = trainx[trainy==0]
trainx = np.concatenate((pos[:65], neg[:65]), axis=0)
trainy = np.array([1]*65+[0]*65).astype(np.uint8)
ind = np.random.permutation(trainx.shape[0])
trainx = trainx[ind]
trainy = trainy[ind]
testx, testy = ddsm_data.load(args.data_dir, subset='test')
nr_batches_train = int(trainx.shape[0] / args.batch_size)
nr_batches_test = int(testx.shape[0] / args.batch_size)
print("the train data is %d"% trainx.shape[0])
print("the test data is %d"% testx.shape[0])
print("compiling......")

disc_layers = [ll.InputLayer(shape=(None, 1, 224, 224))]
disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.1))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (5, 5), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (5, 5), stride=2,pad=1,W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 256, (5, 5),stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 512, (5, 5),stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=sigmoid), train_g=True, init_stdv=0.1))

labels = T.ivector()
images = T.tensor4()
lr = th.shared(np.cast[th.config.floatX](args.learning_rate))
ll.get_output(disc_layers[-1], images, deterministic=False, init=True)
init_updates = [u for l in disc_layers for u in getattr(l, 'init_updates', [])]
output = ll.get_output(disc_layers[-1], images, deterministic=False)
loss = T.mean(lasagne.objectives.binary_crossentropy(predictions=output, targets=labels))
train_acc = T.mean(lasagne.objectives.binary_accuracy(output,labels))
disc_params = ll.get_all_params(disc_layers[-1], trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value()),broadcastable=p.broadcastable) for p in disc_params]
disc_avg_updates = [(a,a+0.01*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[images],outputs=None,updates=init_updates)
train_batch_disc = th.function(inputs=[images,labels],outputs=[loss,train_acc], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[images,labels], outputs=train_acc, givens=disc_avg_givens)
print("Training......")

for epoch in range(num_epoch):
    # if epoch >= decay_epoch:
    #     if epoch % decay_step == 0:
    #         lr.set_value(np.cast[th.config.floatX](lr.get_value()/2))
    begin = time.time()
    if epoch == 0:
        init_param(trainx)
    l = 0.
    tr_err = 0.
    index = 0
    index = 0
    for x_batch, y_batch in datagen.flow(trainx,trainy,batch_size=args.batch_size):
        index += 1
        l_, tr_err_ = train_batch_disc(x_batch, y_batch)
        l += l_
        tr_err += tr_err_
        if index == nr_batches_train:
            break

    l /= nr_batches_train
    tr_err /= nr_batches_train
    te_err = 0.

    for x_batch, y_batch in gen_minibatches(testx, testy, batch_size=args.batch_size):
        te_err += test_batch(x_batch, y_batch)
    te_err /= nr_batches_test
    print("Epoch {}, time = {:.1f}s, train loss = {:.4f}, train acc = {:.4f}, test acc = {:.4f}".format(epoch, time.time()-begin, l, tr_err, te_err))
