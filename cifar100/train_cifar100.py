import argparse
import cPickle
import time
import os
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy,categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
import nn
import sys
import plotting
import seaborn as sns
import cifar100_data
import params
import copy
import scipy.misc
# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='../datasets/')
parser.add_argument('--results_dir', type=str, default='../results/')
parser.add_argument('--count', type=int, default=200)
parser.add_argument('--aug',action='store_true')
args = parser.parse_args()
print(args)


def merge(images, size):
    h, w, c = images[0].shape[0], images[0].shape[1], images[0].shape[2]
    img = np.zeros((size[0] * h, size[1] * w, c))
    for k, image in enumerate(images):
        i = k // size[1]
        j = k % size[1]
        img[i * h:(i + 1) * h, j * h:(j + 1) * h, :] = image
    return img

def gen_minibatches(x,y,batch_size,shuffle=False):
    assert len(x) == len(y), "Training data size don't match"
    if shuffle:
        ids = np.random.permutation(len(x))
    else:
        ids = np.arange(len(x))
    for start_idx in range(0,len(x)-batch_size+1,batch_size):
        ii = ids[start_idx:start_idx+batch_size]
        yield x[ii],y[ii]

## default parameters of ImageDataGenerator
# datagen = ImageDataGenerator(featurewise_center=False,
#                              samplewise_center=False,
#                              featurewise_std_normalization=False,
#                              samplewise_std_normalization=False,
#                              zca_whitening=False,
#                              zca_epsilon=1e-6,
#                              rotation_range=30.,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              shear_range=0.,
#                              zoom_range=0.,
#                              channel_shift_range=0,
#                              fill_mode='nearest',
#                              cval=0.,
#                              horizontal_flip=True,
#                              vertical_flip=True,
#                              rescale=None,
#                              preprocessing_function=None,
#                              data_format=None)

aug_params = dict(rotation_range=30.,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
if args.aug:
    datagen = ImageDataGenerator(**aug_params)
else:
    datagen = ImageDataGenerator()
# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
num_classes = 100
gan_epoch = 100
num_epoch = 300
# decay_epoch = 600
# decay_after_epochs = 5
# learning_rate_decay = 0.90
# load CIFAR-10 and sample 2000 random images

trainx, trainy = cifar100_data.load(args.data_dir, subset='train')
testx, testy = cifar100_data.load(args.data_dir, subset='test')


train = {}
for i in range(num_classes):
    train[i] = trainx[trainy == i][:args.count]
y_data = np.concatenate([trainy[trainy == i][:args.count] for i in range(num_classes)], axis=0)
x_data = np.concatenate([train[i] for i in range(num_classes)], axis=0)
ind = rng.permutation(x_data.shape[0])
trainx = x_data[ind]
trainy = y_data[ind]
print("Training data size = %d"%trainx.shape[0])
nr_batches_train = int(trainx.shape[0] / args.batch_size)
nr_batches_test = int(testx.shape[0] / args.batch_size)

# specify generative model
noise_dim = (args.batch_size, 200)
print("Compiling......")
labels = T.ivector()
x_lab = T.tensor4()
labels_gen = T.ivector()
gen_in_z = ll.InputLayer(shape=noise_dim)
noise = theano_rng.uniform(size=noise_dim)
gen_in_y = ll.InputLayer(shape=(args.batch_size,))
gen_layers = [gen_in_z]
gen_layers.append(nn.MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(ll.DenseLayer(gen_layers[-1], num_units=4 * 4 * 512, W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size, 512, 4, 4)))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 256, 8, 8), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 128, 16, 16), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 3, 32, 32), (5, 5), W=Normal(0.05), nonlinearity=T.tanh))
gen_layers.append(nn.weight_norm(gen_layers[-1], train_g=True, init_stdv=0.1))

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
# disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.2))  #uncomment this line if test without data augmentation
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
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=2 * num_classes, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))

# costs
temp = ll.get_output(disc_layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in disc_layers for u in getattr(l, 'init_updates', [])]
gen_dat = ll.get_output(gen_layers[-1], {gen_in_y: labels_gen, gen_in_z: noise})
output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)
source_lab = T.batched_dot(T.reshape(output_before_softmax_lab, newshape=(args.batch_size, 2, num_classes)), lasagne.utils.one_hot(labels, num_classes).dimshuffle(0, 1, 'x')).dimshuffle(0, 1,)
source_gen = T.batched_dot(T.reshape(output_before_softmax_gen, newshape=(args.batch_size, 2, num_classes)), lasagne.utils.one_hot(labels_gen, num_classes).dimshuffle(0, 1, 'x')).dimshuffle(0, 1,)
class_lab = T.batched_dot(T.reshape(output_before_softmax_lab, newshape=(args.batch_size, 2, num_classes)).dimshuffle(0, 2, 1), T.ones(shape=(args.batch_size, 2, 1))).dimshuffle(0, 1,)
class_gen = T.batched_dot(T.reshape(output_before_softmax_gen, newshape=(args.batch_size, 2, num_classes)).dimshuffle(0, 2, 1), T.ones(shape=(args.batch_size, 2, 1))).dimshuffle(0, 1,)
loss_gen_class = T.mean(categorical_crossentropy(predictions=softmax(class_gen), targets=labels_gen))
loss_gen_source = T.mean(categorical_crossentropy(predictions=softmax(source_gen), targets=T.zeros(shape=(args.batch_size,), dtype='int32')))
loss_lab_class = T.mean(categorical_crossentropy(predictions=softmax(class_lab), targets=labels))
loss_lab_source = T.mean(categorical_crossentropy(predictions=softmax(source_lab), targets=T.zeros(shape=(args.batch_size,), dtype='int32'))) +\
    T.mean(categorical_crossentropy(predictions=softmax(source_gen), targets=T.ones(shape=(args.batch_size,), dtype='int32')))
weight_gen_loss = th.shared(np.float32(0.))
output_lab = ll.get_output(disc_layers[-2],x_lab,deterministic=False)
output_gen = ll.get_output(disc_layers[-2],gen_dat,deterministic=False)
m1 = T.mean(output_lab,axis=0)
m2 = T.mean(output_gen,axis=0)
feature_loss = T.mean(abs(m1-m2))

loss_gen = (1-weight_gen_loss)*(loss_gen_source + 0.5*feature_loss)
loss_lab = (1-weight_gen_loss)*loss_lab_source + weight_gen_loss*(loss_lab_class+loss_gen_class)

#network performance
D_acc_on_real = T.mean(categorical_accuracy(predictions=source_lab, targets=T.zeros(shape=(args.batch_size,))))
D_acc_on_fake = T.mean(categorical_accuracy(predictions=source_gen, targets=T.ones(shape=(args.batch_size,))))
G_acc_on_fake = T.mean(categorical_accuracy(predictions=source_gen, targets=T.zeros(shape=(args.batch_size,))))
performfun = th.function(inputs=[x_lab, labels, labels_gen], outputs=[D_acc_on_real, D_acc_on_fake, G_acc_on_fake])
train_err = T.mean(T.neq(T.argmax(class_lab, axis=1), labels))
# test error
output_before_softmax = ll.get_output(disc_layers[-1], x_lab, deterministic=True)
test_class_lab = T.batched_dot(T.reshape(output_before_softmax, newshape=(args.batch_size, 2, num_classes)).dimshuffle(0, 2, 1), T.ones(shape=(args.batch_size, 2, 1))).dimshuffle(0, 1,)
test_err = T.mean(T.neq(T.argmax(test_class_lab, axis=1), labels))

# Theano functions for training the disc net
learning_rate_var = th.shared(np.cast[th.config.floatX](args.learning_rate))
disc_params = ll.get_all_params(disc_layers[-1], trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab, lr=learning_rate_var, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0. * p.get_value())) for p in disc_params]
disc_avg_updates = [(a, a + 0.01 * (p - a)) for p, a in zip(disc_params, disc_param_avg)]
disc_avg_givens = [(p, a) for p, a in zip(disc_params, disc_param_avg)]
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_lab, labels, labels_gen], outputs=[loss_lab, train_err], updates=disc_param_updates + disc_avg_updates)
test_batch = th.function(inputs=[x_lab, labels], outputs=test_err, givens=disc_avg_givens)
samplefun = th.function(inputs=[labels_gen], outputs=gen_dat)

# Theano functions for training the gen net
gen_params = ll.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=learning_rate_var, mom1=0.5)
train_batch_gen = th.function(inputs=[labels_gen,x_lab], outputs=loss_gen, updates=gen_param_updates)
print("Start training......")
datagen.fit(trainx)
inds = rng.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
# //////////// perform training //////////////
train_list = []
test_list = []
d_acc_on_real_list = []
d_acc_on_fake_list = []
g_acc_on_fake_list = []
for epoch in range(num_epoch):
    if epoch == gan_epoch:
        weight_gen_loss.set_value(np.float32(1.))
    begin = time.time()
    if epoch == 0:
        init_param(trainx[:500]) # data based initialization
    # if (epoch+1) >=decay_epoch:
    #     if (epoch + 1) % decay_after_epochs == 0:
    #         learning_rate_var.set_value(
    #             np.float32(learning_rate_var.get_value() * learning_rate_decay))

    # train
    loss_lab = 0.
    train_err = 0.
    d_acc_on_real = 0.
    d_acc_on_fake = 0.
    g_acc_on_fake = 0.
    index = 0
    if epoch < gan_epoch:
        for x_batch, y_batch in gen_minibatches(trainx, trainy, batch_size=args.batch_size,shuffle=True):
            gen_y = np.int32(np.random.choice(num_classes, (args.batch_size,),replace=False))
            # gen_y = y_batch
            ll, te = train_batch_disc(x_batch, y_batch, gen_y)
            d_acc_on_real_, d_acc_on_fake_, g_acc_on_fake_ = performfun(x_batch,y_batch, gen_y)
            loss_lab += ll
            train_err += te
            d_acc_on_real += d_acc_on_real_
            d_acc_on_fake += d_acc_on_fake_
            g_acc_on_fake += g_acc_on_fake_
            # if (epoch+1) < gan_epoch:
            for rep in range(2):
                # gen_y_ = np.int32(np.random.choice(num_classes, (args.batch_size,)))
                gen_y_ = y_batch
                genloss = train_batch_gen(gen_y_,x_batch)
    else:
        for x_batch, y_batch in datagen.flow(trainx, trainy,batch_size=args.batch_size):
        # for x_batch, y_batch in gen_minibatches(trainx, trainy, batch_size=args.batch_size,shuffle=True):
            index += 1
            gen_y = np.int32(np.random.choice(num_classes, (args.batch_size,),replace=False))
            ll, te = train_batch_disc(x_batch, y_batch, gen_y)
            d_acc_on_real_, d_acc_on_fake_, g_acc_on_fake_ = performfun(x_batch,y_batch, gen_y)
            loss_lab += ll
            train_err += te
            d_acc_on_real += d_acc_on_real_
            d_acc_on_fake += d_acc_on_fake_
            g_acc_on_fake += g_acc_on_fake_
            if index == nr_batches_train:
                break

    loss_lab /= nr_batches_train
    train_err /= nr_batches_train
    d_acc_on_real /= nr_batches_train
    d_acc_on_fake /= nr_batches_train
    g_acc_on_fake /= nr_batches_train
    train_list.append(train_err)
    d_acc_on_real_list.append(d_acc_on_real)
    d_acc_on_fake_list.append(d_acc_on_fake)
    g_acc_on_fake_list.append(g_acc_on_fake)
    # test
    test_err = 0.
    for x_batch, y_batch in gen_minibatches(testx, testy, batch_size=args.batch_size,shuffle=False):
        test_err += test_batch(x_batch, y_batch)
    test_err /= nr_batches_test
    test_list.append(test_err)

    # report
    print("Epoch %d, time = %ds, loss_lab = %.3f, learning_rate = %.6f,train err= %.4f, test err = %.4f" % (epoch, time.time() - begin, loss_lab, learning_rate_var.get_value(),train_err, test_err))
    sys.stdout.flush()

    # generate samples from the model
    sample_y = np.int32(np.random.choice(num_classes, size=(args.batch_size),replace=False))
    sample_x = samplefun(sample_y)
    img_bhwc = np.transpose(sample_x[:100, ], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR-100 samples')
    plotting.plt.savefig(args.results_dir + args.dataset + '/cifar100_sample.png')
    if epoch % 20 == 0:
        NNdiff = np.sum(np.sum(np.sum(np.square(np.expand_dims(sample_x, axis=1) - np.expand_dims(trainx, axis=0)), axis=2), axis=2), axis=2)
        NN = trainx[np.argmin(NNdiff, axis=1)]
        NN = np.transpose(NN[:100], (0, 2, 3, 1))
        NN_tile = plotting.img_tile(NN, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img_tile = np.concatenate((img_tile, NN_tile), axis=1)
        img = plotting.plot_img(img_tile, title='CIFAR-100 samples')
        plotting.plt.savefig(args.results_dir + args.dataset + '/' + str(epoch) + '.png')
    plotting.plt.close('all')

    if epoch % 100 == 0:
        np.savez(args.results_dir + args.dataset + '/disc_params' + str(epoch) + '.npz', *[p.get_value() for p in disc_params])
        np.savez(args.results_dir + args.dataset + '/gen_params' + str(epoch) + '.npz', *[p.get_value() for p in gen_params])
sns.set()
plotting.plt.plot(train_list)
plotting.plt.plot(test_list)
plotting.plt.title('training and testing error')
plotting.plt.ylabel('err')
plotting.plt.xlabel('epoch')
plotting.plt.legend(['training', 'testing'], loc='upper right')
plotting.plt.savefig(args.results_dir + args.dataset + '/error.png')
plotting.plt.clf()
plotting.plt.plot(d_acc_on_real_list)
plotting.plt.plot(d_acc_on_fake_list)
plotting.plt.plot(g_acc_on_fake_list)
plotting.plt.title('network performance')
plotting.plt.ylabel('acc')
plotting.plt.xlabel('epoch')
plotting.plt.legend(['d_acc_on_real', 'd_acc_on_fake', 'g_acc_on_fake'], loc='upper right')
plotting.plt.savefig(args.results_dir + args.dataset + '/performance.png')
np.savez(args.results_dir + args.dataset + '/disc_params' + '.npz', *[p.get_value() for p in disc_params])
np.savez(args.results_dir + args.dataset + '/gen_params' + '.npz', *[p.get_value() for p in gen_params])
