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

from neon.models.model import Model
from neon.backends import gen_backend
from neon.data import ArrayIterator
import numpy as np
import matplotlib.pyplot as plt
import h5py
from my_gan_layers import generator

# backend generation and batch size setting
gen_backend(backend='gpu', batch_size=64)

latent_size = 256
my_generator = Model(generator())
my_generator.load_params('our_gen.prm')

# inference test
#gan.fill_noise(inference_set)
x_new = np.random.randn(100, latent_size)
inference_set = ArrayIterator(X=x_new, make_onehot=False)
test = my_generator.get_outputs(inference_set) # this invokes the model class method that has been modified for this. Find better way.
test = test.reshape((100, 25, 25, 25))
print(test.shape, 'generator output')
plt.plot(test[0, :, 12, :])
plt.savefig('output_img.png')
h5f = h5py.File('output_data.h5', 'w')
h5f.create_dataset('dataset_1', data=test)
