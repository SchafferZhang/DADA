import numpy as np
import os
import theano
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T
import nn
import utils.paramgraphics as paramgraphics
np.set_printoptions(threshold='nan')
seed = 1234
rng = np.random.RandomState(seed)
theano_rng = MRG_RandomStreams(rng.randint(2**15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2**15)))

dim_input = (32,32)
in_channels = 3
colorImg = True
generation_scale = True
n_z = 200
batch_size_g = 1000
num_classes = 100
results_dir = '../results/cifar100'

# symlobs
sym_y_g = T.ivector()
sym_z_input = T.matrix()
sym_z_rand = theano_rng.uniform(size=(batch_size_g,n_z))
sym_z_shared = T.tile(theano_rng.uniform((batch_size_g/num_classes,n_z)), (num_classes,1))

'''models'''
gen_in_z = ll.InputLayer(shape=(batch_size_g, n_z))
gen_in_y = ll.InputLayer(shape=(batch_size_g,))
gen_layers = [gen_in_z]
# gen_layers = [(nn.MoGLayer(gen_in_z, noise_dim=(batch_size_g, n_z)))]
gen_layers.append(nn.MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(ll.DenseLayer(gen_layers[-1], num_units=4 * 4 * 512, W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (batch_size_g, 512, 4, 4)))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (batch_size_g, 256, 8, 8), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (batch_size_g, 128, 16, 16), (5, 5), W=Normal(0.05), nonlinearity=nn.relu))
gen_layers.append(nn.batch_norm(gen_layers[-1], g=None))
gen_layers.append(nn.ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes))
gen_layers.append(nn.Deconv2DLayer(gen_layers[-1], (batch_size_g, 3, 32, 32), (5, 5), W=Normal(0.05), nonlinearity=T.tanh))
gen_layers.append(nn.weight_norm(gen_layers[-1], train_g=True, init_stdv=0.1))
# for layer in gen_layers:
#     print layer.params
#outputs
gen_out_x = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_rand}, deterministic = False)
gen_out_x_shared = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_shared}, deterministic = False)
gen_out_x_interpolation = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_input},deterministic = False)
generate = theano.function(inputs=[sym_y_g], outputs=gen_out_x)
generate_shared = theano.function(inputs=[sym_y_g], outputs=gen_out_x_shared)
generate_interpolation = theano.function(inputs=[sym_y_g,sym_z_input],outputs=gen_out_x_interpolation)

''' load pretrained model '''
gen_params = ll.get_all_params(gen_layers[-1], trainable=True)
f = np.load(results_dir + '/gen_params.npz')
param_values = [f['arr_%d' % i] for i in range(len(f.files))]
for i, p in enumerate(gen_params):
    p.set_value(param_values[i])
    
print("gen_params fed")

# interpolation on latent space z class conditionally

for i in range(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes),batch_size_g/num_classes))
    original_z = np.repeat(rng.uniform(size=(num_classes,n_z)), batch_size_g/num_classes,axis=0)
    target_z = np.repeat(rng.uniform(size=(num_classes,n_z)), batch_size_g/num_classes,axis=0)
    alpha = np.tile(np.arange(batch_size_g/num_classes)*1.0 / (batch_size_g/num_classes -1),num_classes)
    alpha = alpha.reshape(-1,1)
    z = np.float32((1-alpha)*original_z+alpha*target_z)
    x_gen_batch = generate_interpolation(sample_y,z)
    x_gen_batch = x_gen_batch.reshape(batch_size_g,-1)
    image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input,colorImg=colorImg,tile_shape=(batch_size_g/num_classes,num_classes),scale=generation_scale,save_path=os.path.join(results_dir,'interpolation-'+str(i)+'.png'))

for i in range(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    inds = np.random.permutation(batch_size_g)
    sample_y = sample_y[inds]
    x_gen_batch = generate(sample_y)
    x_gen_batch = x_gen_batch.reshape(batch_size_g,-1)
    image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg = colorImg, tile_shape=(batch_size_g/num_classes,num_classes), scale=generation_scale, save_path=os.path.join(results_dir,'random-' + str(i) + '.png'))

for i in range(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    x_gen_batch = generate_shared(sample_y)
    x_gen_batch = x_gen_batch.reshape(batch_size_g, -1)
    image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg = colorImg, tile_shape=(batch_size_g/num_classes,num_classes), scale=generation_scale, save_path=os.path.join(results_dir,'shared-' + str(i) + '.png'))
