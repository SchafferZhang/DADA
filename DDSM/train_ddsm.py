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
import ddsm_data
import nn
import sys
import plotting
import seaborn as sns
import params
import copy
import scipy.misc
from lasagne.layers import set_all_param_values
# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--dataset', type=str, default='DDSM')
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='../datasets')
parser.add_argument('--results_dir', type=str, default='../results/')
parser.add_argument('--aug',action='store_true')
args = parser.parse_args()
print(args)

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

aug_params = dict(rotation_range=5.,zoom_range=0.2,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
if args.aug:
    datagen = ImageDataGenerator(**aug_params)
else:
    datagen = ImageDataGenerator()
# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
num_classes = 2
gan_epoch = 200
num_epoch = 800


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
print("the training data is %d"% trainx.shape[0])
print("the testing data is %d"%testx.shape[0])
# specify generative model
noise_dim = (args.batch_size, 100)
print("Compiling......")
labels = T.ivector()
x_lab = T.tensor4()
labels_gen = T.ivector()
gen_in_z = ll.InputLayer(shape=noise_dim)
noise = theano_rng.uniform(size=noise_dim)
gen_in_y = ll.InputLayer(shape=(args.batch_size,))
gen_layers = [gen_in_z]
gen_layers.append(nn.MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(ll.DenseLayer(gen_layers[-1], num_units=7 * 7 * 512, W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size, 512, 7, 7)))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 512, 14, 14), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 256, 28, 28), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 128, 56, 56), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 128, 112, 112), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size, 1, 224, 224), (5, 5), W=Normal(0.05), nonlinearity=T.tanh))
gen_layers.append(nn.weight_norm(gen_layers[-1], train_g=True, init_stdv=0.1))

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 1, 224, 224))]
disc_layers.append(ll.GaussianNoiseLayer(disc_layers[-1], sigma=0.2))  #uncomment this line if test without data augmentation
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (5, 5), stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (5, 5), stride=2,pad=1,W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 256, (5, 5),stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 512, (5, 5),stride=2, pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=2 * num_classes, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))

# costs
# init_updates = [u for l in disc_layers for u in getattr(l, 'init_updates', [])]
gen_dat = ll.get_output(gen_layers[-1], {gen_in_y: labels_gen, gen_in_z: noise})
output_before_softmax_lab = ll.get_output(disc_layers[-1], x_lab)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat)
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
# output_lab = ll.get_output(disc_layers[-2],x_lab)
# output_gen = ll.get_output(disc_layers[-2],gen_dat)
# m1 = T.mean(output_lab,axis=0)
# m2 = T.mean(output_gen,axis=0)
# feature_loss = T.mean(abs(m1-m2))
loss_gen = (1-weight_gen_loss)*loss_gen_source
# loss_gen = (1-weight_gen_loss)*feature_loss
loss_lab = (1-weight_gen_loss)*loss_lab_source + weight_gen_loss*(loss_lab_class+0.5*loss_gen_class)

#network performance
D_acc_on_real = T.mean(categorical_accuracy(predictions=source_lab, targets=T.zeros(shape=(args.batch_size,))))
D_acc_on_fake = T.mean(categorical_accuracy(predictions=source_gen, targets=T.ones(shape=(args.batch_size,))))
G_acc_on_fake = T.mean(categorical_accuracy(predictions=source_gen, targets=T.zeros(shape=(args.batch_size,))))
performfun = th.function(inputs=[x_lab, labels, labels_gen], outputs=[D_acc_on_real, D_acc_on_fake, G_acc_on_fake])
train_err = T.mean(T.neq(T.argmax(class_lab, axis=1), labels))

# Theano functions for training the disc net
learning_rate_var = th.shared(np.cast[th.config.floatX](args.learning_rate))
disc_params = ll.get_all_params(disc_layers[-1], trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab, lr=learning_rate_var, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0. * p.get_value()),broadcastable=p.broadcastable) for p in disc_params]
disc_avg_updates = [(a, a + 0.01 * (p - a)) for p, a in zip(disc_params, disc_param_avg)]
disc_avg_givens = [(p, a) for p, a in zip(disc_params, disc_param_avg)]
# init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates,on_unused_input='ignore')
train_batch_disc = th.function(inputs=[x_lab, labels, labels_gen], outputs=[loss_lab, train_err], updates=disc_param_updates + disc_avg_updates)
test_batch = th.function(inputs=[x_lab, labels], outputs=train_err, givens=disc_avg_givens)
samplefun = th.function(inputs=[labels_gen], outputs=gen_dat)

# Theano functions for training the gen net
gen_params = ll.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=learning_rate_var, mom1=0.5)
train_batch_gen = th.function(inputs=[labels_gen], outputs=loss_gen, updates=gen_param_updates)
print("Start training......")
datagen.fit(trainx)

# //////////// perform training //////////////
train_list = []
test_list = []
d_acc_on_real_list = []
d_acc_on_fake_list = []
g_acc_on_fake_list = []
for epoch in range(num_epoch):
    # if epoch == 0:
    #     init_param(trainx) # data based initialization
    if epoch == gan_epoch:
        weight_gen_loss.set_value(np.float32(1.))
    begin = time.time()
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
        # for x_batch, y_batch in train_datagen.gen_minibatches(args.batch_size):
            gen_y = np.int32(np.random.choice(num_classes, (args.batch_size,)))
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
                gen_y_ = np.int32(np.random.choice(num_classes, (args.batch_size,)))
                # gen_y_ = y_batch
                genloss = train_batch_gen(gen_y)
    else:
        for x_batch, y_batch in datagen.flow(trainx, trainy,batch_size=args.batch_size):
        # for x_batch, y_batch in gen_minibatches(trainx, trainy, batch_size=args.batch_size,shuffle=True):
            index += 1
            gen_y = np.int32(np.random.choice(num_classes, (args.batch_size,)))
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
    if epoch % 20 == 0:
        sample_y = np.int32(np.random.choice(num_classes, size=(args.batch_size)))
        sample_x = samplefun(sample_y)
        img_bhwc = np.transpose(sample_x[:100, ], (0, 2, 3, 1))
        # print(img_bhwc.shape)
        img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
        # print(img_tile.shape)
        img_tile = np.squeeze(img_tile)
        img = plotting.plot_img(img_tile, title='ddsm samples')
        plotting.plt.savefig(args.results_dir + args.dataset + '/ddsm_sample_'+str(epoch)+'.png')

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
