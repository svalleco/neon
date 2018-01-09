import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback, TrainMulticostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian, Constant
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm, BranchNode, GeneralizedCost
from neon.layers.layer import Linear, Reshape
from neon.layers.container import Tree, Multicost, LayerContainer, GenerativeAdversarial
from neon.models.model import Model
from neon.transforms import Rectlin, Logistic, GANCost, Tanh, MeanSquared
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from neon.layers.layer import Dropout
from neon.data.hdf5iterator import HDF5Iterator
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from neon.optimizers.optimizer import get_param_list
from neon.backends import gen_backend
from neon.backends.backend import Block
from energy_dataset import temp_3Ddata, EnergyData
from my_gan_model import myGAN, myGenerativeAdversarial
from my_gan_layers import discriminator, generator
from sklearn.model_selection import train_test_split
from neon.util.persist import load_obj, save_obj
import h5py
from neon.data import ArrayIterator
from my_gan_control import *

from neon.models.model import Model
from neon.backends import gen_backend
from neon.data import ArrayIterator
import numpy as np
import matplotlib.pyplot as plt
import h5py
from my_gan_layers import generator

# backend generation and batch size setting
gen_backend(backend='gpu', batch_size=mgc_batch_size)

#create a model out of the generator definition
my_generator = Model(generator())

#choose the file from the batch you want to investigate
#generator_filename = '/home/azanetti/goto_CERN_model/results_08-11-2017-00-43_5899196_/5899196_my_gan_model-generator-Epoch 12_[batch_n_18800].prm'
#generator_filename = '/home/azanetti/goto_CERN_model/results_08-11-2017-00-43_5899196_/5899196_my_gan_model-generator-Epoch 12_[batch_n_19600].prm'
generator_filename = '/home/azanetti/goto_CERN_model/results_08-11-2017-00-43_5899196_/5899196_my_gan_model-generator-Epoch 12_[batch_n_19800].prm'


my_generator.load_params(generator_filename)

# set the prefix for interference output
inf_prefix = os.path.basename(generator_filename)

# inference test
#generating input Ep energies, as in fill_noise_sampledE in our GAN model
maxE = 5
minE = 0
myEnergies = np.random.rand(mgc_batch_size) * (maxE - minE) + minE

# random noise with the same distribution used during training
rand_noise = np.random.normal(0, 1, (mgc_batch_size, mgc_latent_size))

# input vector to the generator
x_new = np.zeros((mgc_batch_size, mgc_latent_size))
for i in range(0, mgc_batch_size):
    x_new[i, :] = rand_noise[i, :] * myEnergies[i]

#input iterator to pass to get_outputs
inference_set = ArrayIterator(X=x_new, make_onehot=False)

#submit the noise sample to the generator and get output
gen_out = my_generator.get_outputs(inference_set) # (batch, 15625) this invokes the model class method that has been modified for this. Find better way.
gen_out = gen_out.reshape((mgc_batch_size, 25, 25, 25))

file_name_endings = ["_xz", "_xy", "_yz"]
for b_ind in range(0, mgc_batch_size):
    my_views = [gen_out[b_ind, :, 12, :], gen_out[b_ind, :, :, 12], gen_out[b_ind, 12, :, :]] # 12 to cut the cube in the center where it is supposed to show more energy
    for num_pics, tens_to_pic, f_ending in zip([0, 1, 2], my_views, file_name_endings):
        plt.figure()
        plt.title("Ep submitted for inference:{0:.2f} section{1}\n".format(myEnergies[b_ind], f_ending))
        plt.imshow(tens_to_pic)
        plt.colorbar()
        plt.savefig(mgc_inference_dir + inf_prefix + '_batch_ind_{}_inference_out'.format(b_ind) + f_ending + '.png')
        plt.close()

print(gen_out.shape, 'generator output files saved to dir {} with prefix: {}'.format(mgc_inference_dir, inf_prefix))

h5f = h5py.File(mgc_inference_dir + 'output_data.h5', 'w')
h5f.create_dataset('dataset_1', data=gen_out)
print(" files were saved in {}  \n".format(mgc_inference_dir))

# #test
# import os
# import time
# import datetime
# kappa = 18
# fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')
# fname = os.path.splitext(os.path.basename(__file__))[0] + "-" + time.strftime("%d-%m-%Y-%H-%M-%S") + '_[' + 'kappa_is_{}'.format(kappa) + ']'
# plt_filename = os.path.join(fdir, fname)
# plt.plot(test[0, :, 12, :])
# plt.savefig(plt_filename)
# print(" file {} was saved \n".format(plt_filename))