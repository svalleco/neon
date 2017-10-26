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
from my_gan_costs import RelativeCost

import logging
main_logger = logging.getLogger('neon')
main_logger.setLevel(10)

# backend generation and batch size setting
gen_backend(backend='gpu', batch_size=120)

# load up the data set
#X, y = temp_3Ddata("/home/azanetti/CERNDATA/EGshuffled.h5")
X, y = temp_3Ddata("/data/svalleco/GAN/data/Ele_v1_1_2.h5")
X[X < 1e-6] = 0
#mean = np.mean(X, axis=0, keepdims=True)
#max_elem = np.max(np.abs(X))
#print(np.max(np.abs(X)),'max abs element')
#print(np.min(X),'min element')
# X = (X - mean)/max_elem # commented out as per Sofia suggestion
#print(X.shape, 'X shape')
#print(np.max(X),'max element after normalisation') #not done
#print(np.min(X),'min element after normalisation') #not done
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=42)
print(X_train.shape, 'X train shape')
print(y_train.shape, 'y train shape')




# total epochs of training and size of noise vector to feed the generator
nb_epochs = 30
latent_size = 256

# setup datasets
train_set = EnergyData(X=X_train, Y=y_train, lshape=(1, 25, 25, 25))
# train_set = ArrayIterator(X=X_train, y=y_train, lshape=(1,25,25,25), make_onehot=False)

# grab one iteration from the train_set
iterator = train_set.__iter__()
(X, Y) = iterator.next()
print("printing X and Y")
print X  # this should be shape (N, 25,25, 25)
print Y  # this should be shape (Y1,Y2) of shapes (1)(1)
assert X.is_contiguous
assert Y.is_contiguous
tt = X_train.reshape(X_train.shape[0], 25, 25, 25)
plt.figure()
plt.plot(tt[0, :, 12, :])
my_dir = "results/"
plt.savefig(my_dir + 'example_from_train_set_img_0.png')

plt.figure()
plt.plot(tt[30, :, 12, :])
my_dir = "results/"
plt.savefig(my_dir + 'example_from_train_set_img_30.png')

plt.figure()
plt.plot(tt[50, :, 12, :])
my_dir = "results/"
plt.savefig(my_dir + 'example_from_train_set_img_50.png')

train_set.reset()

# generate test set
valid_set =EnergyData(X=X_test, Y=y_test, lshape=(1,25,25,25))
#valid_set =ArrayIterator(X=X_test, y=y_test, lshape=(1,25,25,25), make_onehot=False)
print 'train_set OK'

my_gen_layers = generator()
my_disc_layers = discriminator()
print my_disc_layers
print my_gen_layers
layers = myGenerativeAdversarial(generator=my_gen_layers, discriminator=my_disc_layers)
print 'layers defined'
print layers

# setup optimizer
#optimizer = GradientDescentMomentum(learning_rate=1e-3, momentum_coef = 0.9)
#optimizer = RMSProp()
optimizer = Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.999, epsilon=1e-8)

# setup cost functions
cost = Multicost(costs=[GeneralizedGANCost(costfunc=GANCost(func="wasserstein")),
#cost = Multicost(costs=[GeneralizedGANCost(costfunc=GANCost(func="modified")), #wasserstein  / modified
                        GeneralizedCost(costfunc=RelativeCost()),
                        GeneralizedCost(costfunc=RelativeCost())])
# cost = Multicost(costs=[GeneralizedGANCost(costfunc=GANCost(func="wasserstein")),
#                         GeneralizedCost(costfunc=RelativeCost()),
#                         GeneralizedCost(costfunc=RelativeCost())])

# initialize model
noise_dim = (latent_size)
gan = myGAN(layers=layers, noise_dim=noise_dim, dataset=train_set, k=5, wgan_param_clamp=0.9,wgan_train_sched=True) # try with k > 1 (=5)

# configure callbacks
callbacks = Callbacks(gan, eval_set=valid_set)
callbacks.add_callback(TrainMulticostCallback())
# fdir = ensure_dirs_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/'))
# fname = os.path.splitext(os.path.basename(__file__))[0] +\
#     '_[' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ']'
# im_args = dict(filename=os.path.join(fdir, fname), hw=32,
#                num_samples=64, nchan=1, sym_range=True)
# callbacks.add_callback(GANPlotCallback(**im_args))
# callbacks.add_save_best_state_callback("./best_state.pkl")

print 'starting training'
# run fit
gan.fit(train_set, num_epochs=nb_epochs, optimizer=optimizer,
        cost=cost, callbacks=callbacks)

# saving parameters using service function from myGan class has issues and temporarily using Model
#gan.save_params('our_gan.prm')

import time
timestamp = time.strftime("%d-%m-%Y-%H-%M-%S")

fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')
generator_file_name = os.path.splitext(os.path.basename(__file__))[0] + "-generator-" + timestamp + '].prm'
discriminator_file_name = os.path.splitext(os.path.basename(__file__))[0] + "-discriminator-" + timestamp + '].prm'

my_generator = Model(gan.layers.generator)
my_generator.save_params(generator_file_name)
my_discriminator = Model(gan.layers.discriminator)
my_discriminator.save_params(discriminator_file_name)

