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
from neon.util.persist import load_obj, save_obj

from energy_dataset import temp_3Ddata, EnergyData
from gan_defs import myGenerativeAdversarial, myGAN
from gan_layers import G_layers, D_layers

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
# new definitions

import logging
main_logger = logging.getLogger('neon')
main_logger.setLevel(10)

gen_backend(backend='cpu', batch_size=64)

# load up the data set
fileName = "/Users/svalleco/GAN/data/Ele_v1_total1.h5"
X, y = temp_3Ddata(fileName)
X[X < 1e-6] = 0 #remove unphysical values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=42)


# setup datasets
train_set = EnergyData(X=X_train, Y=y_train, lshape=(1,25,25,25))

# grab one iteration from the train_set
iterator = train_set.__iter__()
(X, Y) = iterator.next()
print("printing X and Y")
print X  # this should be shape (N, 25,25, 25)
print Y  # this should be shape (Y1,Y2) of shapes (1)(1)
train_set.reset()

# generate test set
valid_set =EnergyData(X=X_test, Y=y_test, lshape=(1,25,25,25))

print 'train_set OK'


# G_layers and D_layers are now Tree containers and not Sequential!!
layers = myGenerativeAdversarial(generator=G_layers, discriminator=D_layers)
                               #discriminator=Sequential(D_layers, name="Discriminator"))
# setup optimizer
optimizer = GradientDescentMomentum(learning_rate=1e-3, momentum_coef = 0.9)

# setup cost function as Binary CrossEntropy
cost = Multicost(costs=[GeneralizedGANCost(costfunc=GANCost(func="wasserstein")),
                        GeneralizedCost(costfunc=MeanSquared()),
                        GeneralizedCost(costfunc=MeanSquared())])
nb_epochs = 1
latent_size = 256
# initialize model
noise_dim = (latent_size)
gan = myGAN(layers=layers, noise_dim=noise_dim, dataset=train_set, k=1, wgan_param_clamp=0.9)

# configure callbacks
callbacks = Callbacks(gan, eval_set=valid_set)
callbacks.add_callback(TrainMulticostCallback())
# fdir = ensure_dirs_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/'))
# fname = os.path.splitext(os.path.basename(__file__))[0] +\
#     '_[' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ']'
# im_args = dict(filename=os.path.join(fdir, fname), hw=32,
#                num_samples=64, nchan=1, sym_range=True)
# callbacks.add_callback(GANPlotCallback(**im_args))
#callbacks.add_save_best_state_callback("./best_state.pkl")

print 'starting training'
# run fit
gan.fit(train_set, num_epochs=nb_epochs, optimizer=optimizer,
        cost=cost, callbacks=callbacks)

my_generator = Model(gan.layers.generator)
my_generator.save_params('our_gen.prm')
my_discriminator = Model(gan.layers.discriminator)
my_discriminator.save_params('our_disc.prm')

