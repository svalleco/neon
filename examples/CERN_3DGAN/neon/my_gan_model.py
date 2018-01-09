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
from my_gan_control import *
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
    and Discriminator stacks as tree containers.

    Arguments:
        layers (list): A list containing two tree containers
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
            self.be.fill_normal(z, mean=0.00311173243963, stdv=0.1) #input noise of same dimensionality of x
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

        myEnergies = np.random.rand(z.shape[1]) * (maxE - minE) + minE
        myEnergies = self.be.array(myEnergies)

        # conditioning of inputs
        for i in range(z.shape[1]):
            z[:, i] = z[:, i] * myEnergies[i]

        # todo: myEnergies should be a class attribute, for which I allocate space on the device prior to this

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
        # todo: review this configuration above that looks very bad
        # in original GAN class is like follows. Review why this was changed
        # # Propagate shapes through the layers to configure
        # prev_input = self.layers.configure(self.noise_dim)
        #
        # if cost is not None:
        #     cost.initialize(prev_input)
        #     self.cost = cost

        # Now allocate space
        self.layers.generator.allocate(accumulate_updates=False)
        self.layers.discriminator.allocate(accumulate_updates=True)
        self.layers.allocate_deltas(None)
        self.initialized = True

        self.y_out = self.be.iobuf((1,))  # realfake
        self.y_Ep = self.be.iobuf((1,))  # Particle Energy
        self.y_SUMEcal = self.be.iobuf((1,))  # Total Energy on Cal

        self.zbuf = self.be.iobuf(self.noise_dim)
        self.ybuf = self.be.iobuf((1,))
        self.z0 = self.be.iobuf(self.noise_dim)  # a fixed noise buffer for generating images
        self.fill_noise_sampledE(self.z0, normal=(self.noise_type == 'normal'))

        # buffers for costs
        self.cost_dis = np.empty((1,), dtype=np.float32)
        self.cost_dis_Ep = np.empty((1,), dtype=np.float32)
        self.cost_dis_SUMEcal = np.empty((1,), dtype=np.float32)
        self.cost_gen = np.empty((1,), dtype=np.float32)

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

    def plot_partials_generations(self, Gen_output, cond_labels, prob_fake_real, Ep_estimated, SUMEcal_estimated,\
                                  kind="generated"):
        # setting filenames
        fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), my_gan_results_dir)
        plfname = mgc_run_random_prefix + os.path.splitext(os.path.basename(__file__))[0] + "-" + \
                       mgc_timestamp +  "-" + kind + "-" + 'Epoch {}'.format(self.epoch_index) + '_[' + \
                           'batch_n_{}'.format(self.current_batch) + ']'
        h5fname = mgc_run_random_prefix + os.path.splitext(os.path.basename(__file__))[0] + "-" + \
                      mgc_timestamp + 'Epoch {}'.format(self.epoch_index) + '_[' + \
                          'batch_n_{}'.format(self.current_batch) + '].h5'
        plt_filename = os.path.join(fdir, plfname)
        h5_filename = os.path.join(fdir, h5fname)

        # plotting
        Gen_output_3D = Gen_output.reshape((25, 25, 25, self.be.bsz))
        my_views = [Gen_output_3D[:, 12, :, 0], Gen_output_3D[:, :, 12, 0], Gen_output_3D[12, :, :, 0]]
        # 12 here is to get the central slice of the cube

        file_name_endings = ["_xz", "_xy", "_yz"]
        for num_pics, tens_to_pic, f_ending in zip([0, 1, 2], my_views, file_name_endings):
            plt.figure()
            plt.title("Ep_r:{0:.2f} R/F:{1:.2f} Ep_e:{2:.2f} Ecal_e:{3:.2f}\n".format(\
                          cond_labels[0, 0], prob_fake_real[0, 0], Ep_estimated[0, 0], SUMEcal_estimated[0, 0]))
            if mgc_plot_matrix:
                plt.imshow(tens_to_pic)
                plt.colorbar()
            else:
                plt.plot(tens_to_pic)
            plt.savefig(plt_filename + f_ending)
            plt.close()
            print("PARTIAL IMAGE Gen_output {} was saved".format(plt_filename + f_ending))

        # saving to hdf5 file the total output tensor from the generator
        if mgc_save_training_progress:
            h5f = h5py.File(h5_filename, 'w')
            h5f.create_dataset('dataset_1', data=Gen_output_3D)
            print("PARTIAL HDF5 file of Gen_output --------file {} was saved\n".format(h5_filename))

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
        y_data = y_noise = self.y_out
        y_data_Ep = y_noise_Ep = self.y_Ep
        y_data_SUMEcal = y_noise_SUMEcal = self.y_SUMEcal

        # iterate through minibatches of the dataset
        for mb_idx, (x, labels, mb_max, mb_mean) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)

            # clip all discriminator parameters to a cube in case of WGAN
            if self.wgan_param_clamp:
                self.clip_param_in_layers(self.layers.discriminator.layers_to_optimize,
                                          self.wgan_param_clamp)

            # prepare buffers
            self.cost_dis[:] = 0
            self.cost_gen[:] = 0
            self.cost_dis_Ep[:] = 0
            self.cost_dis_SUMEcal[:] = 0

            # 1 - TRAIN DISCRIMINATOR ON NOISE
            print("\n\nRUN ID: {0}\n>> START MINIBATCH {1}\n>> Step 1 - TRAIN DISCR. ON NOISE: {2}-th time".
                  format(mgc_run_random_prefix, self.current_batch, self.current_batch))

            my_energies = self.fill_noise_sampledE(z, normal=(self.noise_type == 'normal'))
            Gz = self.fprop_gen(z)
            # Gz is a list, with 1 element of batchsize size, being the generator defined as tree
            (y_noise[:], y_noise_SUMEcal[:], y_noise_Ep[:]) = self.fprop_dis(Gz[0])

            if mgc_debug:
                assert not np.isnan(z.get()).any(), "P1 - NAN in z input of generator"
                try:
                    assert not np.isnan(Gz[0].get()).any(), "P2 - NAN in Gz output of generator"
                except:
                    print(Gz[0].get())
                assert not np.isnan(y_noise.get()).any(), "P3 - NAN in y_noise"
                assert not np.isnan(y_noise_Ep.get()).any(), "P3 - NAN in y_noise_Ep"
                try:
                    assert not np.isnan(y_noise_SUMEcal.get()).any(), "P4 - NAN in y_noise_SUMEcal"
                except:
                    print(y_noise_SUMEcal.get())
            if mgc_print_tensor_examples:
                print("\n1 - TRAIN DISCRIMINATOR ON NOISE: example of estimated R/F of generated images (y_noise.get()):")
                print(y_noise.get())
                print("\n1 - TRAIN DISCRIMINATOR ON NOISE: example of estimated Ep of generated images (y_noise_Ep.get()):")
                print(y_noise_Ep.get())
                print("\n1 - TRAIN DISCRIMINATOR ON NOISE: example of estimated SUMECal of generated images (y_noise_SUMEcal.get()):")
                print(y_noise_SUMEcal.get())

            # buffering some values used later
            y_temp[:] = y_noise  # Real/Fake discr output on noise: that is, probability that the input was real
            rand_Ep = my_energies[0, :]  # Ep inputs used to condition the noise input to the generator

            # computing derivatives of cost function wrt discriminator outputs, on all output lines
            # approx of Esum as 2 times Ep, numerically, after rescaling in energy_dataset.py
            delta_noise = self.cost.costs[0].costfunc.bprop_noise(y_noise)
            delta_noise_Ep = self.cost.costs[2].costfunc.bprop(y_noise_Ep, rand_Ep)
            if mgc_data_normalization == "for_tanh_output":
                tp = (2 * rand_Ep - mb_mean * Gz[0].shape[0]) / mb_max
                tpval = self.be.empty((1, self.be.bsz))  # allocate space for output
                tpval[:] = tp  # execute the op-tree
            elif mgc_data_normalization == "for_logistic_output":
                tpval = 2 * rand_Ep / mb_max
            else:
                tpval = 2 * rand_Ep
            delta_noise_SUMEcal = self.cost.costs[1].costfunc.bprop(y_noise_SUMEcal, tpval)

            # computing gradient contributions from all three output lines, for discriminator weights
            self.bprop_dis([delta_noise, delta_noise_SUMEcal, delta_noise_Ep])

            # set flag for accumulation of gradients so far computed as addition of the gradients on
            # data training is next
            self.layers.discriminator.set_acc_on(True)

            # 2 - TRAIN DISCRIMINATOR ON DATA
            print(">> Step 2 - TRAIN DISCR. ON DATA for the {0}-th time".format(self.current_batch))
            # getting separate discriminator outputs.
            (y_noise[:], y_noise_SUMEcal[:], y_noise_Ep[:]) = self.fprop_dis(x)

            if mgc_debug:
                assert not np.isnan(y_noise.get()).any(), "P5 - not a number in y_noise"
                assert not np.isnan(y_noise_Ep.get()).any(), "P5 - not a number in y_noise_Ep"
                assert not np.isnan(y_noise_SUMEcal.get()).any(), "P5 - not a number in y_noise_SUMEcal"
            if mgc_print_tensor_examples:
                print("\n2 - TRAIN DISCRIMINATOR ON DATA: example of Ep from dataset (labels[0, :].get())")
                print(labels[0, :].get())
                print("\n2 - TRAIN DISCRIMINATOR ON DATA: example of SUMEcal from dataset (labels[1, :].get())")
                print(labels[1, :].get())
                print("\n2 - TRAIN DISCRIMINATOR ON DATA: example of R/F estimated (y_data.get())")
                print(y_data.get())
                print("\n2 - TRAIN DISCRIMINATOR ON DATA: example of model estimated Ep (y_data_Ep.get()):")
                print(y_data_Ep.get())
                print("\n2 - TRAIN DISCRIMINATOR ON DATA: example of model estimated SUMECal (y_data_SUMEcal.get()):")
                print(y_data_SUMEcal.get())
                print("\n2 - TRAIN DISCRIMINATOR ON DATA: example of numpy sum of one image data from dataset: np.sum(x.get()[:], axis=(0,))")
                temsum = np.sum(x.get()[:], axis=(0,))
                print(temsum)

            # computing derivatives of cost function wrt discriminator outputs, on all output lines
            delta_data = self.cost.costs[0].costfunc.bprop_data(y_data)
            delta_data_SUMEcal = self.cost.costs[1].costfunc.bprop(y_data_SUMEcal, labels[1, :])
            delta_data_Ep = self.cost.costs[2].costfunc.bprop(y_data_Ep, labels[0, :])

            # plotting data for checking consistency
            if self.current_batch % mgc_data_saving_freq == 0 \
                    and mgc_print_image_of_training_on_data:
                self.plot_partials_generations(x.get(), labels[0, :].get(), y_data.get(), y_data_Ep.get(),
                                               y_data_SUMEcal.get(), kind="data")

            # computing gradient contributions from all three output lines, for discriminator weights
            self.bprop_dis([delta_data, delta_data_SUMEcal, delta_data_Ep])

            # using gradients so calculated to tweak the discriminator weights
            self.optimizer.optimize(self.layers.discriminator.layers_to_optimize, epoch=epoch)

            # gradient accumulation flag off for discr
            self.layers.discriminator.set_acc_on(False)

            # keep GAN cost values for the current minibatch
            # abuses get_cost(y,t) using y_noise as the "target"
            self.cost_dis[:] = self.cost.costs[0].get_cost(y_data, y_temp, cost_type='dis')

            if mgc_compute_all_costs:
                self.cost_dis_SUMEcal[:] = self.cost.costs[1].get_cost(y_data_SUMEcal[0], labels[1, :][0])
                self.cost_dis_Ep[:] = self.cost.costs[2].get_cost(y_data_Ep[0], labels[0, :][0])
                print("END OF DISCRMINATOR TRAINING: COSTS: Real/Fake cost: {}    Ep cost: {}     SUMEcal cost: {}".format\
                          (self.cost_dis[0], self.cost_dis_Ep[0], self.cost_dis_SUMEcal[0]))
                # TODO: why computing generator cost affect the cost displayed...
                # TODO: ...by the ProgressBar callback, actually written by TrainCostCallback?
                #self.cost_gen[:] = self.cost.costs[0].get_cost(y_data, y_temp, cost_type='gen')

            # 3 - TRAIN GENERATOR
            if mgc_train_gen:
                if self.current_batch == self.last_gen_batch + self.get_k(self.gen_iter):
                    print("---> 3 - TRAIN GENERATOR {}-th time".format(self.gen_iter))

                    # gradient accumulation flag off for generator
                    self.layers.generator.set_acc_on(True)

                    # generator can trained more times in a row
                    ntimes_train_gen = mgc_gen_times if (not mgc_cost_function == "Wasserstein") else 1
                    for i in range(ntimes_train_gen):
                        my_energies = self.fill_noise_sampledE(z, normal=(self.noise_type == 'normal'))
                        Gz = self.fprop_gen(z)
                        (y_noise[:], y_noise_SUMEcal[:], y_noise_Ep[:]) = self.fprop_dis(Gz[0])
                        t = my_energies[0, :]

                        if mgc_debug:
                            assert not np.isnan(Gz[0].get()).any(), "P6 - not a number in Gz output of generator"

                         #printing tensors for investigation
                        if mgc_print_tensor_examples:
                            print(
                            "\n3 - TRAIN GENERATOR: example of estimated Ep of generated images (y_noise_Ep.get()):")
                            print(y_noise_Ep.get())
                            print(
                            "\n3 - TRAIN GENERATOR: example of estimated SUMECal of generated images (y_noise_SUMEcal.get()):")
                            print(y_noise_SUMEcal.get())

                        # computing derivatives of cost functions wrt discriminator outputs, on all output lines
                        # approx of Esum as 2 times Ep, numerically, after rescaling in energy_dataset.py
                        # attempt to do the trick, like switching labels
                        if mgc_trick:
                            delta_noise = self.cost.costs[0].costfunc.bprop_data(y_noise)
                        else:
                            delta_noise = self.cost.costs[0].costfunc.bprop_noise(y_noise)
                        delta_noise_Ep = self.cost.costs[2].costfunc.bprop(y_noise_Ep, t)
                        if mgc_data_normalization == "for_tanh_output":
                            tp = (2 * t - mb_mean * Gz[0].shape[0])/mb_max
                            tpval = self.be.empty((1, self.be.bsz))  # allocate space for output
                            tpval[:] = tp  # execute the op-tree
                        elif mgc_data_normalization == "for_logistic_output":
                            tpval = 2 * t / mb_max
                        else:
                            tpval = 2 * t
                        delta_noise_SUMEcal = self.cost.costs[1].costfunc.bprop(y_noise_SUMEcal, tpval)

                        # discriminator and generator backprop: computing gradient contributions
                        # from all three output lines, for discriminator weights
                        delta_nnn = self.bprop_dis([delta_noise, delta_noise_SUMEcal, delta_noise_Ep])
                        self.bprop_gen(delta_nnn)

                    # using gradients so calculated to tweak the generator weights
                    self.optimizer.optimize(self.layers.generator.layers_to_optimize, epoch=epoch)

                    # gradient accumulation flag off for generator
                    self.layers.generator.set_acc_on(False)

                    # keep GAN cost values for the current minibatch
                    # add something like this with support in callbacks for generator cost displaying?
                    self.cost_dis[:] = self.cost.costs[0].get_cost(y_temp, y_noise, cost_type='dis')
                    if mgc_compute_all_costs:
                        self.cost_dis_Ep[:] = self.cost.costs[1].get_cost(y_noise_Ep[0], t)
                        self.cost_dis_SUMEcal[:] = self.cost.costs[2].get_cost(y_noise_SUMEcal[0], tpval)
                        self.cost_gen[:] = self.cost.costs[0].get_cost(y_data, y_noise, cost_type='gen')
                        print("END OF GENERATOR TRAINING: COSTS: Discr. Real/Fake cost: {}    Ep cost: {}     SUMEcal cost: {}  Generator cost: {}".format\
                                  (self.cost_dis[0], self.cost_dis_Ep[0], self.cost_dis_SUMEcal[0], self.cost_gen[0]))

                    if mgc_print_tensor_examples:
                        print("\n3 - TRAIN GENERATOR: Estimated SUM \n")
                        print(y_noise_SUMEcal.get())
                        print("\n3 - TRAIN GENERATOR: tpval")
                        tpvall = self.be.empty((1, self.be.bsz))  # allocate space for output
                        tpvall[:] = tpval  # execute the op-tree
                        print(tpvall.get())

                    # temporary saving of plots and hdf5 data:
                    if self.current_batch % mgc_data_saving_freq == 0:
                        # last output of the generator from the backend object
                        self.plot_partials_generations(Gz[0].get(), t.get(), y_noise.get(), y_noise_Ep.get(),
                                                       y_noise_SUMEcal.get(), kind="generated")

                    # accumulate total cost.
                    self.total_cost[:] = self.total_cost + self.cost_dis
                    self.last_gen_batch = self.current_batch
                    self.gen_iter += 1

            #saving params: todo: move this saving into callbacks
            if mgc_save_prm:
                fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), my_gan_results_dir)
                genfname = mgc_run_random_prefix + os.path.splitext(os.path.basename(__file__))[0] + "-generator-" + \
                          'Epoch {}'.format(self.epoch_index) + '_[' + 'batch_n_{}'.format(self.current_batch) + '].prm'
                discfname = mgc_run_random_prefix + os.path.splitext(os.path.basename(__file__))[0] + "-discriminator-" + \
                          'Epoch {}'.format(self.epoch_index) + '_[' + 'batch_n_{}'.format(self.current_batch) + '].prm'
                gen_filename = os.path.join(fdir, genfname)
                disc_filename = os.path.join(fdir, discfname)
                my_generator = Model(self.layers.generator)
                my_generator.save_params(gen_filename)
                # my_discriminator = Model(self.layers.discriminator)
                # my_discriminator.save_params(disc_filename)
                print("Saved prm files for generator and :\nGen--> {}\nDisc--> {}".format(gen_filename, disc_filename))

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
        self.data_batch, self.noise_batch = x, self.fprop_gen(self.z0)[0] # our generator is a tree
        pass


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
