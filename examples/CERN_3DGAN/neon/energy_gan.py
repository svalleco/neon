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
import numpy as np
from sklearn.model_selection import train_test_split
from neon.util.persist import load_obj, save_obj
import matplotlib.pyplot as plt
import h5py
# new definitions

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
        myEnergies = np.transpose(np.random.rand(z.shape[1]) * (maxE - minE) + minE)
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

            # train discriminator on noise
            myEnergies = self.fill_noise_sampledE(z, normal=(self.noise_type == 'normal'))
            #z = self.z0
            Gz = self.fprop_gen(z) # Gz is a list qwith 1 element, being the generator defined as tree
            y_noise_list = self.fprop_dis(Gz[0]) # this is due to the generator defined as tree
            y_noise = y_noise_list[0] # getting the Real/Fake
            y_noise_Ep = y_noise_list[1]
            y_noise_SUMEcal = y_noise_list[2]

            y_temp[:] = y_noise

            delta_noise = self.cost.costs[0].costfunc.bprop_noise(y_noise)
            t = myEnergies[0, :]
            delta_noise_Ep = self.cost.costs[1].costfunc.bprop(t , y_noise_Ep)
            delta_noise_SUMEcal = self.cost.costs[2].costfunc.bprop(.5 * t, y_noise_SUMEcal)
            delta_nnn = self.bprop_dis([delta_noise, delta_noise_Ep, delta_noise_SUMEcal])
            # delta_nnn = self.bprop_dis([delta_noise])
            self.layers.discriminator.set_acc_on(True)

            # train discriminator on data: in this case the additional lines will be taken into account
            y_data_list = self.fprop_dis(x)
            y_data = y_data_list[0] # this is due to the generator defined as tree, but this time all
            # the output lines are meaningful
            y_data_Ep = y_data_list[1]
            y_data_SUMEcal = y_data_list[2]
            delta_data = self.cost.costs[0].costfunc.bprop_data(y_data)
            l = labels[0, :]
            delta_data_Ep = self.cost.costs[1].costfunc.bprop(l, y_data_Ep)
            delta_data_SUMEcal = self.cost.costs[2].costfunc.bprop(labels[1, :], y_data_SUMEcal)
            delta_ddd = self.bprop_dis([delta_data, delta_data_Ep, delta_data_SUMEcal])
            self.optimizer.optimize(self.layers.discriminator.layers_to_optimize, epoch=epoch)
            self.layers.discriminator.set_acc_on(False)

            # keep GAN cost values for the current minibatch
            # abuses get_cost(y,t) using y_noise as the "target"
            self.cost_dis[:] = self.cost.costs[0].get_cost(y_data, y_temp, cost_type='dis')

            #console feedback
            #print(" \n minibatch index {}".format(mb_idx))

            # train generator
            if self.current_batch == self.last_gen_batch + self.get_k(self.gen_iter):
                print(" ---> now training the generator {}-th time".format(self.gen_iter))
                self.fill_noise_sampledE(z, normal=(self.noise_type == 'normal'))
                Gz = self.fprop_gen(z)
                y_noise_list = self.fprop_dis(Gz[0])
                y_noise = y_noise_list[0]  # just getting the WGAN cost
                y_temp[:] = y_data
                delta_noise = self.cost.costs[0].costfunc.bprop_generator(y_noise)
                delta_dis = self.bprop_dis([delta_noise])
                self.bprop_gen(delta_dis)
                self.optimizer.optimize(self.layers.generator.layers_to_optimize, epoch=epoch)
                # keep GAN cost values for the current minibatch
                # self.cost_gen[:] = self.cost.costs[0].get_cost(y_data, y_noise, cost_type='gen')
                #  add something like this with support in callbacks for generator cost displaying??
                self.cost_dis[:] = self.cost.costs[0].get_cost(y_temp, y_noise, cost_type='dis')
                # accumulate total cost.
                self.total_cost[:] = self.total_cost + self.cost_dis
                self.last_gen_batch = self.current_batch
                self.gen_iter += 1

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

    def get_generator_outputs(self, dataset):
        """
        Get the activation outputs of the final model layer for the dataset

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform fit on

        Returns:
            Host numpy array: the output of the final layer for the entire Dataset
        """
        self.initialize(dataset)
        dataset.reset()  # Move "pointer" back to beginning of dataset
        n = dataset.nbatches
        x = self.layers.layers[-1].outputs
        assert not isinstance(x, list), "Can not get_outputs with Branch terminal"
        Ypred = None
        for idx, input_data in enumerate(dataset):
            x = self.fprop_gen(input_data[0], inference=True)
            if isinstance(x, list):
                x = x[0]
            if Ypred is None:
                (dim0, dim1) = x.shape
                Ypred = np.empty((n * dim1, dim0), dtype=x.dtype)
                nsteps = dim1 // self.be.bsz
            cur_batch = slice(idx * dim1, (idx + 1) * dim1)
            Ypred[cur_batch] = x.get().T

        # Handle the recurrent case.
        if nsteps != 1:
            b, s = (self.be.bsz, nsteps)
            Ypred = Ypred.reshape((n, s, b, -1)).transpose(0, 2, 1, 3).copy().reshape(n * b, s, -1)

        return Ypred[:dataset.ndata]


# load up the data set
X, y = temp_3Ddata()
X[X < 1e-6] = 0
mean = np.mean(X, axis=0, keepdims=True)
max_elem = np.max(np.abs(X))
print(np.max(np.abs(X)),'max abs element')
print(np.min(X),'min element')
X = (X- mean)/max_elem
print(X.shape, 'X shape')
print(np.max(X),'max element after normalisation')
print(np.min(X),'min element after normalisation')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=0.1, random_state=42)
print(X_train.shape, 'X train shape')
print(y_train.shape, 'y train shape')

gen_backend(backend='gpu', batch_size=64)

#X_train.reshape((X_train.shape[0], 25 * 25 * 25))

# setup datasets
# train_set = EnergyData(X=X_train, Y=y_train, lshape=(1,25,25,25))
from neon.data import ArrayIterator
train_set = ArrayIterator(X=X_train, y=y_train, lshape=(1,25,25,25), make_onehot=False)

# grab one iteration from the train_set
iterator = train_set.__iter__()
(X, Y) = iterator.next()
print("printing X and Y")
print X  # this should be shape (N, 25,25, 25)
print Y  # this should be shape (Y1,Y2) of shapes (1)(1)
assert X.is_contiguous
assert Y.is_contiguous
train_set.reset()

# generate test set
#valid_set =EnergyData(X=X_test, Y=y_test, lshape=(1,25,25,25))
valid_set =ArrayIterator(X=X_test, y=y_test, lshape=(1,25,25,25), make_onehot=False)

print 'train_set OK'
#tate=lt.plot(X_train[0, 12])
#plt.savefigure('data_img.png')

# setup weight initialization function
init = Gaussian(scale=0.01)

# discriminiator using convolution layers
lrelu = Rectlin(slope=0.1)  # leaky relu for discriminator
# sigmoid = Logistic() # sigmoid activation function
conv1 = dict(init=init, batch_norm=False, activation=lrelu, bias=init)
conv2 = dict(init=init, batch_norm=False, activation=lrelu, padding=2, bias=init)
conv3 = dict(init=init, batch_norm=False, activation=lrelu, padding=1, bias=init)
b1 = BranchNode("b1")
b2 = BranchNode("b2")
branch1 = [b1,
            Conv((5, 5, 5, 32), **conv1),
            Dropout(keep = 0.8),
            Conv((5, 5, 5, 8), **conv2),
            BatchNorm(),
            Dropout(keep = 0.8),
            Conv((5, 5, 5, 8), **conv2),
            BatchNorm(),
            Dropout(keep = 0.8),
            Conv((5, 5, 5, 8), **conv3),
            BatchNorm(),
            Dropout(keep = 0.8),
            Pooling((2, 2, 2)),
            Affine(1024, init=init, activation=lrelu),
            BatchNorm(),
            Affine(1024, init=init, activation=lrelu),
            BatchNorm(),
            b2,
            Affine(nout=1, init=init, bias=init, activation=Logistic())
            ] #real/fake
branch2 = [b2, 
           Affine(nout=1, init=init, bias=init, activation=lrelu)] #E primary
branch3 = [b1,
           Linear(1, init=Constant(val=1.0))] #SUM ECAL

D_layers = Tree([branch1, branch2, branch3], name="Discriminator") #keep weight between branches equal to 1. for now (alphas=(1.,1.,1.) as by default )

# generator using convolution layers
init_gen = Gaussian(scale=0.001)
relu = Rectlin(slope=0)  # relu for generator
pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
str1 = dict(str_h=2, str_w=2, str_d=2)
conv1 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad1, strides=str1, bias=init_gen)
pad2 = dict(pad_h=2, pad_w=2, pad_d=2)
str2 = dict(str_h=2, str_w=2, str_d=2)
conv2 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad2, strides=str2, bias=init_gen)
pad3 = dict(pad_h=0, pad_w=0, pad_d=0)
str3 = dict(str_h=1, str_w=1, str_d=1)
conv3 = dict(init=init_gen, batch_norm=False, activation=Tanh(), padding=pad3, strides=str3, bias=init_gen)
bg = BranchNode("bg")
branchg  = [bg,
            Affine(1024, init=init_gen, bias=init_gen, activation=relu),
            BatchNorm(),
            Affine(8 * 7 * 7 * 7, init=init_gen, bias=init_gen),
            Reshape((8, 7, 7, 7)),
            Deconv((6, 6, 6, 6), **conv1), #14x14x14
            BatchNorm(),
            # Linear(5 * 14 * 14 * 14, init=init),
            # Reshape((5, 14, 14, 14)),
            Deconv((5, 5, 5, 64), **conv2), #27x27x27
            BatchNorm(),
            Conv((3, 3, 3, 1), **conv3)
           ]

G_layers = Tree([branchg], name="Generator")

print D_layers
print G_layers

# G_layers and D_layers are now Tree containers and not Sequential!!
layers = myGenerativeAdversarial(generator=G_layers, discriminator=D_layers)
                               #discriminator=Sequential(D_layers, name="Discriminator"))
print 'layers defined'
print layers
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

#gan.save_params('our_gan.prm')

inference_set = train_set #HDF5Iterator(x_new, None, nclass=2, lshape=(latent_size))

x_new = np.random.randn(100, latent_size)
inference_set = ArrayIterator(X=x_new, make_onehot=False)
my_generator = Model(gan.layers.generator)
my_generator.save_params('our_gen.prm')
my_discriminator = Model(gan.layers.discriminator)
my_discriminator.save_params('our_disc.prm')
#gan.fill_noise(inference_set)
test = my_generator.get_outputs(inference_set)
test = np.float32(test*max_elem + mean)
test = test.reshape((100, 25, 25, 25))

print(test.shape, 'generator output')

plt.plot(test[0, :, 12, :])
plt.savefig('output_img.png')

h5f = h5py.File('output_data.h5', 'w')
h5f.create_dataset('dataset_1', data=test)

