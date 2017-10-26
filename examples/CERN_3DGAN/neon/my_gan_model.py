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
from my_gan_layers import discriminator, generator
from sklearn.model_selection import train_test_split
from neon.util.persist import load_obj, save_obj
import h5py
from neon.data import ArrayIterator
from neon.layers.container import Tree, Multicost, LayerContainer, GenerativeAdversarial
from neon.models.model import Model
from neon.optimizers.optimizer import get_param_list
from neon.backends.backend import Block
import numpy as np
from neon.util.persist import load_obj, save_obj
import matplotlib.pyplot as plt
import os
import time
import h5py

import logging
main_logger = logging.getLogger('neon')
main_logger.setLevel(10)

class myGenerativeAdversarial(GenerativeAdversarial): # LayerContainer):
    """
    Container for Generative Adversarial Net (GAN). It contains the Generator
    and Discriminator stacks as sequential containers.

    Arguments:
        layers (list): A list containing two Sequential containers
    """
    def __init__(self, generator, discriminator, name=None):
        super(LayerContainer, self).__init__(name)

        self.generator = generator
        self.discriminator = discriminator
        self.layers = self.generator.layers + self.discriminator.layers

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """
        padstr = '\n' + '  ' * level
        ss = '  ' * level + self.classnm + padstr
        ss += '  ' * level + 'Generator:\n'
        ss += padstr.join([l.nested_str(level + 1) for l in self.generator.layers])
        ss += '\n' + '  ' * level + 'Discriminator:\n'
        ss += padstr.join([l.nested_str(level + 1) for l in self.discriminator.layers])
        return ss

    def get_terminal(self):
        return self.generator.get_terminal() + self.discriminator.get_terminal()


class myGAN(Model):
    """
    Model for Generative Adversarial Networks.

    Arguments:
        layers: Generative Adversarial layer container
        noise_dim (Tuple): Dimensionality of the noise feeding the generator
        noise_type (Str): Noise distribution, 'normal' (default) or 'uniform'
        weights_only (bool): set to True if you do not want to recreate layers
                             and states during deserialization from a serialized model
                             description.  Defaults to False.
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule for updating
                               model parameters (i.e., GradientDescentMomentum, Adadelta)
        k (int): Number of data batches per noise batch
        wgan_param_clamp (float or None): In case of WGAN weight clamp value, None for others
        wgan_train_sched (bool): Whether to use the FAIR WGAN training schedule of critics
    """
    def __init__(self, layers, noise_dim, dataset, noise_type='normal', weights_only=False,
                 name="model", optimizer=None, k=1,
                 wgan_param_clamp=None, wgan_train_sched=False):
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.k = k
        self.wgan_param_clamp = wgan_param_clamp
        self.wgan_train_sched = wgan_train_sched
        self.nbatches = 0
        self.ndata = 0
        super(myGAN, self).__init__(layers, weights_only=weights_only, name=name,
                                  optimizer=optimizer)

    @staticmethod
    def clip_param_in_layers(layer_list, abs_bound=None):
        """
        Element-wise clip all parameter tensors to between
        ``-abs_bound`` and ``+abs_bound`` in a list of layers.

        Arguments:
            layer_list (list): List of layers
            be (Backend object): Backend in which the tensor resides
            abs_bound (float, optional): Value to element-wise clip gradients
                                         or parameters. Defaults to None.
        """
        param_list = get_param_list(layer_list)
        for (param, grad), states in param_list:
            if abs_bound:
                param[:] = param.backend.clip(param, -abs(abs_bound), abs(abs_bound))

    def fill_noise(self, z, normal=True):
        """
        Fill z with either uniform or normally distributed random numbers
        """
        if normal:
            # Note fill_normal is not deterministic
            self.be.fill_normal(z)
        else:
            z[:] = 2 * self.be.rand() - 1.

    def fill_noise_sampledE(self, z, normal=True, minE=0., maxE=5.):
        """
        Fill z with either uniform or normally distributed random numbers
        """
        if normal:
            # Note fill_normal is not deterministic
            self.be.fill_normal(z)
        else:
            z[:] = 2 * self.be.rand() - 1.
        #z[:] = z[:]*(self.be.rand()*(maxE -minE)+minE)
        myEnergies = np.random.rand(z.shape[1]) * (maxE - minE) + minE
        myEnergies = self.be.array(myEnergies)
        for i in range (z.shape[1]):
            z[:, i] = z[:, i] * myEnergies[i]
        return myEnergies.T

    def initialize(self, dataset, cost=None):
        """
        Propagate shapes through the layers to configure, then allocate space.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform initialization on
            cost (Cost): Defines the function which the model is minimizing based
                         on the output of the last layer and the input labels.
        """
        if self.initialized:
            return

        # Propagate shapes through the layers to configure
        prev_input = dataset
        prev_input = self.layers.generator.configure(self.noise_dim)
        prev_input = self.layers.discriminator.configure(dataset)
        # prev_input = self.layers.configure(self.noise_dim)

        if cost is not None:
            cost.initialize(self.layers)
            self.cost = cost

        # Now allocate space
        self.layers.generator.allocate(accumulate_updates=False)
        self.layers.discriminator.allocate(accumulate_updates=True)
        self.layers.allocate_deltas(None)
        self.initialized = True
        self.zbuf = self.be.iobuf(self.noise_dim)
        self.ybuf = self.be.iobuf((1,))
        self.z0 = self.be.iobuf(self.noise_dim)  # a fixed noise buffer for generating images
        self.fill_noise_sampledE(self.z0, normal=(self.noise_type == 'normal'))
        self.cost_dis = np.empty((1,), dtype=np.float32)
        self.current_batch = self.gen_iter = self.last_gen_batch = 0

    def get_k(self, giter):
        """
        WGAN training schedule for generator following Arjovsky et al. 2017

        Arguments:
            giter (int): Counter for generator iterations
        """
        if self.wgan_train_sched and (giter < 25 or giter % 500 == 0):
            return 100
        else:
            return self.k

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost[:] = 0
        last_gen_iter = self.gen_iter
        z, y_temp = self.zbuf, self.ybuf

        # iterate through minibatches of the dataset
        for mb_idx, (x, labels) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)

            # clip all discriminator parameters to a cube in case of WGAN
            if self.wgan_param_clamp:
                self.clip_param_in_layers(self.layers.discriminator.layers_to_optimize,
                                          self.wgan_param_clamp)

            # TRAIN DISCRIMINATOR ON NOISE
            myEnergies = self.fill_noise_sampledE(z, normal=(self.noise_type == 'normal'))
            Gz = self.fprop_gen(z) # Gz is a list with 1 element of batchsize size, being the generator defined as tree
            y_noise_list = self.fprop_dis(Gz[0]) # this is due to the generator defined as tree

            # getting separate discriminator outputs
            y_noise = y_noise_list[0] # getting the Real/Fake
            y_noise_Ep = y_noise_list[1] # getting Particle Energy
            y_noise_SUMEcal = y_noise_list[2] # getting Total Energy on Cal

            # buffering fake/real discriminator output
            y_temp[:] = y_noise

            # computing derivatives of cost function wrt discriminator outputs, on all output lines
            delta_noise = self.cost.costs[0].costfunc.bprop_noise(y_noise)
            t = myEnergies[0, :]
            delta_noise_Ep = self.cost.costs[1].costfunc.bprop(t, y_noise_Ep)
            # approx of Esum as 2 times Ep, numerically, after rescaling in energy_dataset.py
            delta_noise_SUMEcal = self.cost.costs[2].costfunc.bprop(2 * t, y_noise_SUMEcal)

            # computing gradient contributions from all three output lines, for discriminator weights
            delta_nnn = self.bprop_dis([delta_noise, delta_noise_Ep, delta_noise_SUMEcal])
            # delta_nnn = self.bprop_dis([delta_noise]) # for backprop of fake/real line only

            # set flag for accumulation of gradients so far computed as addition of the gradients on
            # data training is next
            self.layers.discriminator.set_acc_on(True)

            # TRAIN DISCRIMINATOR ON DATA
            y_data_list = self.fprop_dis(x)

            # getting separate discriminator outputs
            y_data = y_data_list[0] # this is due to the generator defined as tree
            y_data_Ep = y_data_list[1] # getting Particle Energy
            y_data_SUMEcal = y_data_list[2] # getting Total Energy on Cal

            # computing derivatives of cost function wrt discriminator outputs, on all output lines
            delta_data = self.cost.costs[0].costfunc.bprop_data(y_data)
            delta_data_Ep = self.cost.costs[1].costfunc.bprop(labels[0, :], y_data_Ep)
            delta_data_SUMEcal = self.cost.costs[2].costfunc.bprop(labels[1, :], y_data_SUMEcal)

            # computing gradient contributions from all three output lines, for discriminator weights
            delta_ddd = self.bprop_dis([delta_data, delta_data_Ep, delta_data_SUMEcal])

            # using gradients so calculated to tweak the discriminator weights
            self.optimizer.optimize(self.layers.discriminator.layers_to_optimize, epoch=epoch)

            # gradient accumulation flag off
            self.layers.discriminator.set_acc_on(False)

            # keep GAN cost values for the current minibatch
            # abuses get_cost(y,t) using y_noise as the "target"
            self.cost_dis[:] = self.cost.costs[0].get_cost(y_data, y_temp, cost_type='dis') ##<<<--- review how to disc cost is computed here

            # TRAIN GENERATOR
            if self.current_batch == self.last_gen_batch + self.get_k(self.gen_iter):
                print(" ---> now training the generator {}-th time".format(self.gen_iter))
                for i in range(2): # testing as per Sofia's request 2 iterations on generator
                    myEnergies = self.fill_noise_sampledE(z, normal=(self.noise_type == 'normal'))
                    Gz = self.fprop_gen(z)
                    y_noise_list = self.fprop_dis(Gz[0])

                    # getting separate discriminator outputs
                    y_noise = y_noise_list[0]  # getting the Real/Fake
                    y_noise_Ep = y_noise_list[1]  # getting Particle Energy
                    y_noise_SUMEcal = y_noise_list[2]  # getting Total Energy on Cal

                    y_temp[:] = y_data

                    # computing derivatives of cost function wrt discriminator outputs, on all output lines
                    delta_noise = self.cost.costs[0].costfunc.bprop_noise(y_noise)
                    t = myEnergies[0, :]
                    delta_noise_Ep = self.cost.costs[1].costfunc.bprop(t, y_noise_Ep)
                    # approx of Esum as 2 times Ep, numerically, after rescaling in energy_dataset.py
                    delta_noise_SUMEcal = self.cost.costs[2].costfunc.bprop(2 * t, y_noise_SUMEcal)

                    # computing gradient contributions from all three output lines, for discriminator weights
                    delta_nnn = self.bprop_dis([delta_noise, delta_noise_Ep, delta_noise_SUMEcal])
                    #delta_dis = self.bprop_dis([delta_noise])
                    self.bprop_gen(delta_nnn)
                    self.layers.generator.set_acc_on(True)

                self.layers.generator.set_acc_on(False)
                self.optimizer.optimize(self.layers.generator.layers_to_optimize, epoch=epoch)
                # keep GAN cost values for the current minibatch
                # self.cost_gen[:] = self.cost.costs[0].get_cost(y_data, y_noise, cost_type='gen')
                #  add something like this with support in callbacks for generator cost displaying??
                self.cost_dis[:] = self.cost.costs[0].get_cost(y_temp, y_noise, cost_type='dis') ##<<<--- review how to disc cost is computed here
                # accumulate total cost.
                self.total_cost[:] = self.total_cost + self.cost_dis
                self.last_gen_batch = self.current_batch
                self.gen_iter += 1

            #adding brutal temporary saving of plots and hdf5 data
            if self.current_batch % 50 == 0:
                # last output of the generator from the backend object (tested with GPU)
                Gen_output = Gz[0].get()

                #filenames
                timestamp = time.strftime("%d-%m-%Y-%H-%M")
                fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')
                plfname = os.path.splitext(os.path.basename(__file__))[0] + "-" + timestamp + \
                          '_[' + 'batch_n_{}'.format(self.current_batch) + ']'
                h5fname = os.path.splitext(os.path.basename(__file__))[0] + "-" + timestamp + \
                          '_[' + 'batch_n_{}'.format(self.current_batch) + '].h5'
                plt_filename = os.path.join(fdir, plfname)
                h5_filename = os.path.join(fdir, h5fname)

                #plotting
                Gen_output = Gen_output.T # in model.get_outputs(self, dataset), used in inference script, result of fprop is transposed...
                Gen_output = Gen_output.reshape((self.be.bsz, 0, 25, 25, 25)) #64 but test with 128
                plt.figure()
                plt.plot(Gen_output[0, 0, :, 12, :])
                plt.savefig(plt_filename)
                print("\nPARTIAL IMAGE Gen_output[0, :, 12, :] --------file {} was saved".format(plt_filename))

                #saving to hdf5 file the total output tensor from the generator
                h5f = h5py.File(h5_filename, 'w')
                h5f.create_dataset('dataset_1', data=Gen_output)
                print("PARTIAL HDF5 file of Gen_output --------file {} was saved\n".format(h5_filename))

            self.be.end(Block.minibatch, mb_idx)
            callbacks.on_minibatch_end(epoch, mb_idx)
            self.current_batch += 1

        # now we divide total cost by the number of generator iterations,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on the generator
        assert self.gen_iter > last_gen_iter, \
            "at least one generator iteration is required for total cost estimation in this epoch"
        self.total_cost[:] = self.total_cost / (self.gen_iter - last_gen_iter)

        # Package a batch of data for plotting
        self.data_batch, self.noise_batch = x, self.fprop_gen(self.z0)

    def fprop_gen(self, x, inference=False):
        """
        fprop the generator layer stack
        """
        return self.layers.generator.fprop(x, inference)

    def fprop_dis(self, x, inference=False):
        """
        fprop the discriminator layer stack
        """
        return self.layers.discriminator.fprop(x, inference)

    def bprop_dis(self, delta):
        """
        bprop the discriminator layer stack
        """
        return self.layers.discriminator.bprop(delta)

    def bprop_gen(self, delta):
        """
        bprop the generator layer stack
        """
        return self.layers.generator.bprop(delta)

    def save_params(self, param_path, keep_states=True):
        """
        Serializes and saves model parameters to the path specified.

        Arguments:
            param_path (str): File to write serialized parameter dict to.
            keep_states (bool): Whether to save optimizer states too.
                                Defaults to True.
        """
        self.serialize(keep_states=keep_states, fn=param_path)

    # serialize tells how to write out the parameters we've learned so
    # far and associate them with layers. it can ignore layers with no
    # learned parameters. the model stores states to pass to the
    # optimizers.  if we're saving the model out for inference, we
    # don't need to remember states.
    def serialize(self, fn=None, keep_states=True):
        """
        Creates a dictionary storing the layer parameters and epochs complete.

        Arguments:
            fn (str): file to save pkl formatted model dictionary
            keep_states (bool): Whether to save optimizer states.

        Returns:
            dict: Model data including layer parameters and epochs complete.
        """

        # get the model dict with the weights
        pdict = self.get_description(get_weights=True, keep_states=keep_states)
        pdict['epoch_index'] = self.epoch_index + 1
        if self.initialized:
            if not hasattr(self.layers, 'decoder'):
                pdict['train_input_shape'] = self.layers.in_shape
            else:
                # serialize shapes both for encoder and decoder
                pdict['train_input_shape'] = (self.layers.encoder.in_shape +
                                              self.layers.decoder.in_shape)
        if fn is not None:
            save_obj(pdict, fn)
            return
        return pdict
