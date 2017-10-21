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
from gen_data_norm import gen_rhs
from neon.backends import gen_backend
from neon.backends.backend import Block
from energy_dataset import temp_3Ddata, EnergyData
from my_gan_model import myGAN, myGenerativeAdversarial
from my_gan_layers import discriminator, generator
import numpy as np
from sklearn.model_selection import train_test_split
from neon.util.persist import load_obj, save_obj
import matplotlib.pyplot as plt
import h5py
from neon.data import ArrayIterator
from neon.models.model import Model
from neon.backends import gen_backend
from neon.data import ArrayIterator
from energy_dataset import temp_3Ddata, EnergyData
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
from my_gan_layers import generator

# backend generation and batch size setting
gen_backend(backend='gpu', batch_size=64)

latent_size = 256

# load up the data set
X, y = temp_3Ddata("/home/azanetti/CERNDATA/EGshuffled.h5")
X[X < 1e-6] = 0
mean = np.mean(X, axis=0, keepdims=True)
max_elem = np.max(np.abs(X))
print(np.max(np.abs(X)),'max abs element')
print(np.min(X),'min element')
X = (X - mean)/max_elem
print(X.shape, 'X shape')
print(np.max(X),'max element after normalisation')
print(np.min(X),'min element after normalisation')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=0.1, random_state=42)
print(X_train.shape, 'X train shape')
print(y_train.shape, 'y train shape')
#
#  setup datasets
train_set = EnergyData(X=X_train, Y=y_train, lshape=(1, 25, 25, 25))


my_generator = Model(generator())
my_generator.load_params('our_gen.prm')

# inference test
#gan.fill_noise(inference_set)
inference_set = train_set #HDF5Iterator(x_new, None, nclass=2, lshape=(latent_size))
x_new = np.random.randn(100, latent_size)
inference_set = ArrayIterator(X=x_new, make_onehot=False)
test = my_generator.get_outputs(inference_set) # this invokes the model class method that has been modified for this. Find better way.
test = test.reshape((100, 25, 25, 25))
print(test.shape, 'generator output')
plt.plot(test[0, :, 12, :])
plt.savefig('output_img.png')
h5f = h5py.File('output_data.h5', 'w')
h5f.create_dataset('dataset_1', data=test)
