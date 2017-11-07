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
gen_backend(backend='gpu', batch_size=my_gan_control_batch_size)

latent_size = 256
my_generator = Model(generator())
#choose the file from the batch you want to investigate
# 3795296_my_gan_model-generator-Epoch 0_[batch_n_80].prm
generator_filename = '/home/azanetti/CERN/neon/examples/CERN_3DGAN/neon/results_08-11-2017-00-21_3795296_/3795296_my_gan_model-generator-Epoch 0_[batch_n_80].prm'
my_generator.load_params(generator_filename)
inf_prefix = os.path.basename(generator_filename)

# inference test
#gan.fill_noise(inference_set)
x_new = np.random.randn(my_gan_control_batch_size, my_gan_control_latent_size)
inference_set = ArrayIterator(X=x_new, make_onehot=False)

#submit the noise sample to the generator
test = my_generator.get_outputs(inference_set) # this invokes the model class method that has been modified for this. Find better way.
test = test.reshape((my_gan_control_batch_size, 25, 25, 25))
print(test.shape, 'generator output')
plt.plot(test[0, :, 12, :])
my_dir = "inference_results/"
plt.savefig(my_dir + inf_prefix + '_inference_out.png')
h5f = h5py.File(my_dir + 'output_data.h5', 'w')
h5f.create_dataset('dataset_1', data=test)
print(" files were saved in {}  \n".format(my_dir))

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