import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback, TrainMulticostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian, Constant
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm, BranchNode, GeneralizedCost
from neon.layers.layer import Linear, Reshape
from neon.layers.container import Tree, Multicost, LayerContainer, GenerativeAdversarial
from neon.models.model import Model
from neon.transforms import Rectlin, Logistic, GANCost, Tanh, MeanSquared, SumSquared
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from neon.layers.layer import Dropout
from neon.data.hdf5iterator import HDF5Iterator
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from neon.optimizers.optimizer import get_param_list, MultiOptimizer, Optimizer
#from gen_data_norm import gen_rhs
from neon.backends import gen_backend
from neon.backends.backend import Block
from energy_dataset import temp_3Ddata, EnergyData, my_gan_HDF5Iterator
from my_gan_model import myGAN, myGenerativeAdversarial
from my_gan_layers import discriminator, generator
import numpy as np
from sklearn.model_selection import train_test_split
from neon.util.persist import load_obj, save_obj
import matplotlib.pyplot as plt
import h5py
from neon.data import ArrayIterator
from my_gan_costs_and_optimizer import RelativeCost, DummyOptimizer
from my_gan_control import *

import logging

def print_figure(my_tensor, filename, Ep):
    plt.figure()
    plt.title("Ep:{0:.2f}".format(Ep))
    if plot_matrix:
        plt.imshow(my_tensor)
        plt.colorbar()
    else:
        plt.plot(my_tensor)
    plt.savefig(res_dir + my_run_random_prefix + filename)

def main():
    # my code here
    main_logger = logging.getLogger('neon')
    main_logger.setLevel(10)

    # backend generation and batch size setting
    batch_size = my_gan_control_batch_size
    gen_backend(backend='gpu', batch_size=batch_size)

    # total epochs of training and size of noise vector to feed the generator
    nb_epochs = my_gan_control_nb_epochs
    latent_size = my_gan_control_latent_size

    data_filename = "/home/azanetti/CERNDATA/Ele_v1_1_2.h5"
    # load up the data set
    if my_use_hdf5_iterator:
        print("Using my_gan_HDF5 Loader")
        train_set = my_gan_HDF5Iterator(data_filename)
        valid_set = my_gan_HDF5Iterator(data_filename) # not provided nby the hdf5 file so for now like this...

    else:
        print("Using EnergyData Loader")
        AllRealImages, AllLabels = temp_3Ddata(data_filename)
        #X, y = temp_3Ddata("/data/svalleco/GAN/data/Ele_v1_1_2.h5")

        # AllRealImages[AllRealImages < 1e-6] = 0
        # mean = np.mean(AllRealImages, axis=0, keepdims=True)
        # max_elem = np.max(np.abs(AllRealImages))
        # AllRealImages = (AllRealImages - mean)/max_elem

        # X_train, X_test, y_train, y_test = train_test_split(AllRealImages, AllLabels, train_size=0.9, test_size=0.1, random_state=42)
        # print(X_train.shape, 'X train shape')
        # print(y_train.shape, 'y train shape')

        # setup datasets
        # train_set = EnergyData(X=X_train, Y=y_train, lshape=(1, 25, 25, 25))
        # valid_set = EnergyData(X=X_test, Y=y_test, lshape=(1, 25, 25, 25))

        train_set = EnergyData(X=AllRealImages, Y=AllLabels, lshape=(1, 25, 25, 25))
        valid_set = EnergyData(X=AllRealImages, Y=AllLabels, lshape=(1, 25, 25, 25))

    # grab one iteration from the train_set
    iterator = train_set.__iter__()
    (X, Y, _, _) = iterator.next()
    print("printing X and Y")
    print X
    print Y
    assert X.is_contiguous
    assert Y.is_contiguous

    # plotting out some pictures from the database
    filename_list = ['example_from_train_set_img_0_xz.png', 'example_from_train_set_img_0_xy.png', 'example_from_train_set_img_0_yz.png']
    tt = X.get()
    yy = Y.get()
    yy[ yy < 1e-6 ] = 0#removing non physical values
    in_batch = np.random.randint(0, batch_size)
    lyr = np.random.randint(0, 25)
    tt = tt.reshape(25, 25, 25, batch_size)
    tensor_to_print = [tt[:, lyr, :, in_batch], tt[:, :, lyr, in_batch], tt[lyr, :, :, in_batch]]
    for i in range(3):
        print_figure(tensor_to_print[i], filename_list[i], yy[0, in_batch])

    # resetting train_set
    train_set.reset()
    print 'train_set OK'

    my_gen_layers = generator()
    my_disc_layers = discriminator()
    layers = myGenerativeAdversarial(generator=my_gen_layers, discriminator=my_disc_layers)

    if my_debug:
        print my_disc_layers
        print my_gen_layers
        print("Generator option is: {}".format(generator_option))
        print 'layers defined'
        print layers

    # setup optimizer for generator:
    learning_rate = my_gan_control_LR_generator
    if my_gan_control_generator_optimizer == "Adam":
        my_gen_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    elif my_gan_control_generator_optimizer == "RMSProp":
        my_gen_optimizer = RMSProp() #learning_rate=learning_rate)
    else:
        my_gen_optimizer = GradientDescentMomentum(learning_rate=learning_rate, momentum_coef=0.9, gradient_clip_value = 5)
    print("Optimizer in use for Generator is: {}".format(my_gen_optimizer.get_description()))

    #setup optimizer for discriminator
    learning_rate = my_gan_control_LR_discriminator
    if my_gan_control_discriminator_optimizer == "Adam":
        my_discr_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-8)
    elif my_gan_control_discriminator_optimizer == "RMSProp":
        my_discr_optimizer = RMSProp()  # learning_rate=learning_rate)
    else:
        my_discr_optimizer = GradientDescentMomentum(learning_rate=learning_rate, momentum_coef=0.9, gradient_clip_value=5)
    print("Optimizer in use for Discriminator is: {}".format(my_discr_optimizer.get_description()))

    # optimizer mapping to layers
    mapping = {'NotOptimizeLinear': DummyOptimizer(), 'Discriminator': my_discr_optimizer, 'default': my_gen_optimizer}
    optimizer = MultiOptimizer(mapping)

    # setup cost functions R/F, SUMEcal, Ep
    if my_control_cost_function == "Wasserstein":
        my_func = "wasserstein"
    elif my_control_cost_function == "Modified":
        my_func = "modified"
    else:
        my_func = "original"
    if my_gan_control_relative_vs_meansquared == "MeanSquared":
        cost = Multicost(costs=[GeneralizedGANCost(costfunc=GANCost(func=my_func)), #wasserstein / modified /original
                            GeneralizedCost(costfunc=MeanSquared()),
                            GeneralizedCost(costfunc=MeanSquared())])
        print("Using MeanSquared cost function for Auxiliary Classifiers")
    else: #RelativeCost
        cost = Multicost(costs=[GeneralizedGANCost(costfunc=GANCost(func=my_func)),  # wasserstein / modified /original
                                GeneralizedCost(costfunc=RelativeCost()), #RelativeCost() / MeanSquared()
                                GeneralizedCost(costfunc=RelativeCost())])
        print("Using RelativeCost cost function for Auxiliary Classifiers")

    # initialize model
    noise_dim = (latent_size,)
    gan = myGAN(layers=layers, noise_dim=noise_dim, dataset=train_set, k=my_gan_k, wgan_param_clamp=my_gan_control_param_clamp) #, wgan_param_clamp=0.9,wgan_train_sched=True) # try with k > 1 (=5)

    # configure callbacks
    #callbacks = Callbacks(gan, eval_set=valid_set)
    callbacks = Callbacks(gan, output_file= res_dir + my_run_random_prefix + "callbacks_out_" + timestamp + '.h5')
    callbacks.add_callback(TrainMulticostCallback()) # position in the list of callbacks can be specified here. Put 0?

    #training
    print 'starting training'
    gan.fit(train_set, num_epochs=nb_epochs, optimizer=optimizer,
            cost=cost, callbacks=callbacks)

    # saving parameters using service function from myGan class has issues and temporarily using Model
    #gan.save_params('our_gan.prm')

    fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), res_dir)
    generator_file_name = my_run_random_prefix + os.path.splitext(os.path.basename(__file__))[0] + "-generator-" + timestamp + '].prm'
    discriminator_file_name = my_run_random_prefix +  os.path.splitext(os.path.basename(__file__))[0] + "-discriminator-" + timestamp + '].prm'

    my_generator = Model(gan.layers.generator)
    my_generator.save_params(generator_file_name)
    my_discriminator = Model(gan.layers.discriminator)
    my_discriminator.save_params(discriminator_file_name)

if __name__ == "__main__":
    main()