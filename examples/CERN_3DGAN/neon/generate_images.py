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
from neon.data.dataiterator import ArrayIterator
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from neon.optimizers.optimizer import get_param_list
from gen_data_norm import gen_rhs
from neon.backends import gen_backend
from neon.backends.backend import Block
from energy_dataset import temp_3Ddata, EnergyData
import numpy as np
from sklearn.model_selection import train_test_split
from neon.util.persist import load_obj, save_obj
import matplotlib.pyplot as plt
import h5py
# new definitions
from gan_defs import myGenerativeAdversarial, myGAN
from gan_layers import G_layers, D_layers
from energy_dataset import temp_3Ddata, EnergyData
import logging
main_logger = logging.getLogger('neon')
main_logger.setLevel(10)





# G_layers and D_layers are now Tree containers and not Sequential!!
layers = myGenerativeAdversarial(generator=G_layers, discriminator=D_layers)
print 'layers defined'

latent_size = 256

print 'starting inference'

my_generator = Model(G_layers)
my_generator.load_params('our_gen.prm')

x_new = np.random.randn(1200, latent_size) 
inference_set = ArrayIterator(x_new, make_onehot=False)

test = my_generator.get_outputs(inference_set)

print(test.shape, 'generator output')

test =  test.reshape((1200, 25, 25, 25))
#storing images as ECAL
fakeHCAL = np.zeros((n_events,5,5,60))
fakeTARGET = np.ones((n_events,1,5))

h5f = h5py.File('output_data.h5', 'w')
h5f.create_dataset('ECAL', data=test)
h5f.create_dataset('HCAL',data=fakeHCAL)
h5f.create_dataset('target',data=fakeTARGET)
