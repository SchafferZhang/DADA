from __future__ import print_function
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
import nn
import cifar10_data
import svhn_data
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='../datasets/')
parser.add_argument('--results_dir', type=str, default='../results/')
parser.add_argument('--count', type=int, default=200)
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

aug_params = dict(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
if args.aug:
    datagen = ImageDataGenerator(**aug_params)
else:
    datagen = ImageDataGenerator()
num_classes = 10
num_epoch = 600

rng = np.random.RandomState(seed=1)

if args.dataset == 'cifar10':
    trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
    testx, testy = cifar10_data.load(args.data_dir, subset='test')
elif args.dataset == 'svhn':
    trainx, trainy = svhn_data.load(args.data_dir,'train')
    testx, testy = svhn_data.load(args.data_dir,'test')
train = {}
for i in range(10):
    train[i] = trainx[trainy == i][:args.count]
y_data = np.concatenate([trainy[trainy == i][:args.count] for i in range(10)], axis=0)
x_data = np.concatenate([train[i] for i in range(10)], axis=0)
ind = rng.permutation(x_data.shape[0])
trainx = x_data[ind]
trainy = y_data[ind]
nr_batches_train = int(trainx.shape[0] / args.batch_size)
nr_batches_test = int(testx.shape[0] / args.batch_size)
print("the training data is %d"% trainx.shape[0])

#classificaiton model
print("compiling......")
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3, 3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3, 3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3, 3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3, 3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3, 3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3, 3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3, 3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
# disc_layers.append(nn.MinibatchLayer(disc_layers[-1], num_kernels=100))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=2*num_classes, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))


labels = T.ivector()
images = T.tensor4()
lr = th.shared(np.cast[th.config.floatX](args.learning_rate))
ll.get_output(disc_layers[-1], images, deterministic=False, init=True)
init_updates = [u for l in disc_layers for u in getattr(l, 'init_updates', [])]
output = ll.get_output(disc_layers[-1], images, deterministic=False)
class_lab = T.batched_dot(T.reshape(output, newshape=(args.batch_size, 2, num_classes)).dimshuffle(0, 2, 1), T.ones(shape=(args.batch_size, 2, 1))).dimshuffle(0, 1,)
output_deter = ll.get_output(disc_layers[-1], images, deterministic=True)
test_class_lab = T.batched_dot(T.reshape(output_deter, newshape=(args.batch_size, 2, num_classes)).dimshuffle(0, 2, 1), T.ones(shape=(args.batch_size, 2, 1))).dimshuffle(0, 1,)
loss = T.mean(lasagne.objectives.categorical_crossentropy(predictions=lasagne.nonlinearities.softmax(class_lab), targets=labels))
train_err = T.mean(T.neq(T.argmax(class_lab,axis=1),labels))
test_err = T.mean(T.neq(T.argmax(test_class_lab,axis=1),labels))
disc_params = ll.get_all_params(disc_layers[-1], trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[images],outputs=None,updates=init_updates)
train_batch_disc = th.function(inputs=[images,labels],outputs=[loss,train_err], updates=disc_param_updates+disc_avg_updates)
test_batch = th.function(inputs=[images,labels], outputs=test_err, givens=disc_avg_givens)

datagen.fit(trainx)
print("Training......")
for epoch in range(num_epoch):
    # if epoch >= decay_epoch:
    #     if epoch % decay_step == 0:
    #         lr.set_value(np.cast[th.config.floatX](lr.get_value()/2))
    begin = time.time()
    if epoch == 0:
        init_param(trainx[:500])
    l = 0.
    tr_err = 0.
    index = 0
    for x_batch, y_batch in datagen.flow(trainx, trainy, batch_size=args.batch_size):
    # for x_batch, y_batch in gen_minibatches(trainx, trainy, args.batch_size, shuffle=True):
        index += 1
        train_batch_disc(x_batch, y_batch)
        if index == nr_batches_train:
            break
    for x_batch, y_batch in gen_minibatches(trainx, trainy, args.batch_size,shuffle=False):
        tr_err += test_batch(x_batch, y_batch)
    tr_err /= nr_batches_train

    te_err = 0.
    for t in range(nr_batches_test):
        te_err += test_batch(testx[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    te_err /= nr_batches_test

    print("Epoch {}, time = {:.1f}s, train err = {:.4f}, test err = {:.4f}".format(epoch, time.time()-begin, tr_err, te_err))



