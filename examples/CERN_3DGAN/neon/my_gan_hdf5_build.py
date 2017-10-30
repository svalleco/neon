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
from my_gan_control import *

import logging
main_logger = logging.getLogger('neon')
main_logger.setLevel(10)

# backend generation and batch size setting
batch_size = 128
gen_backend(backend='gpu', batch_size=128)

# load up the data set
AllRealImages, AllLabels = temp_3Ddata("/home/azanetti/CERNDATA/Ele_v1_1_2.h5")
#X, y = temp_3Ddata("/data/svalleco/GAN/data/Ele_v1_1_2.h5")

AllRealImages[AllRealImages < 1e-6] = 0
# mean = np.mean(AllRealImages, axis=0, keepdims=True)
# max_elem = np.max(np.abs(AllRealImages))
# AllRealImages = (AllRealImages - mean)/max_elem # commented out as per Sofia suggestion

X_train, X_test, y_train, y_test = train_test_split(AllRealImages, AllLabels, train_size=0.9, test_size=0.1, random_state=42)
print(X_train.shape, 'X train shape')
print(y_train.shape, 'y train shape')

# total epochs of training and size of noise vector to feed the generator
nb_epochs = 30
latent_size = 256

########## below hyp of generating new hdf5 corresponding to request but no space on disk
# filename_str = "/home/azanetti/CERNDATA/Ele_v1_1_2-for-neon"
#
# # generate the HDF5 file
# datsets = {'train': (X_train, y_train),
#            'test': (X_test, y_test)}
#
# for ky in ['train', 'test']:
#     df = h5py.File(filename_str + '_{}.h5'.format(ky), 'w')
#
#     # input images
#     in_dat = datsets[ky][0]
#     df.create_dataset('input', data=in_dat)
#     df['input'].attrs['lshape'] = (1, 25, 25, 25)  # (C, H, W, D)
#
#     # can also add in a mean image or channel by channel mean for color image
#     # for mean subtraction during data iteration
#     # e.g.
#     if ky == 'train':
#         mean_image = np.mean(X_train, axis=0)
#     # use training set mean for both train and val data sets
#     df.create_dataset('mean', data=mean_image)
#
#     target = datsets[ky][1].reshape((-1, 1))  # make it a 2D array
#     df.create_dataset('output', data=target)
#     df['output'].attrs['nclass'] = 10
#     df.close()



class my_gan_HDF5Iterator(ArrayIterator):
    """
    Data iterator which uses an HDF5 file as the source of the data, useful when
    the entire HDF5 dataset cannot fit into memory (for smaller datasets, use the ArrayIterator).

    To initialize the HDF5Iterator, simply call::

        train_set = HDF5Iterator('your_data_path.h5')

    The HDF5 file format must contain the following datasets:

    - `input` (ndarray):
                        Input data, which is a 2-D array (float or uint8) that
                        has size `(N, F)`, where `N` is the number of examples,
                        and `F` is the number of features. For images, `F = C*H*W` where
                        `C` is the number of channels and `H` and `W` are the height
                        and width of the image, respectively. This data must also
                        have the following attributes:
        - `lshape` (tuple):
                    Tuple of ints indicating the shape of each
                    input (for examples, image data may have
                    an lshape of `[C, H, W]`)
        - `mean` (ndarray):
                    The mean to subtract, either formatted as (F, 1) or
                    a mean for each channel with dimensions (C, 1)
    - `output` (ndarray):
                        An optional dataset which, if supplied, will be
                        used at the target/expected output of the network. the
                        array should have the shape `(N, M)` where `N` is the number
                        of items (must match the `N` dim of the input set)
                        and `M` is the size of the output data which must match
                        size of ouput from the output layer of the network.

    For cases where the output should be converted to a one-hot encoding (see Loading Data),
    use the `HDF5IteratorOneHot`. Or for autoencoder problems, use `HDFIteratorAutoencoder`.
    """
    def __init__(self, hdf_filename, name=None):
        """
        Args:
            hdf_filename (string): Path to the HDF5 datafile.
            name (string, optional): Name to assign this iterator. Defaults to None.
        """
        super(ArrayIterator, self).__init__(name=name)

        self.hdf_filename = hdf_filename

        if not os.path.isfile(hdf_filename):
            raise IOError('File not found %s' % hdf_filename)
        self.hdf_file = h5py.File(hdf_filename, mode='r', driver=None)

        # input data array
        self.inp = self.hdf_file['Ecal']
        self.ndata = self.inp.shape[0]

        # must have at least 1 minibatch of data in the file
        assert self.ndata >= self.be.bsz
        self.start = 0

        # the input array unflattened size
        self.lshape = my_gan_lshape # originally: tuple(self.inp.attrs['lshape'])
        self.shape = self.lshape

        if 'output' in self.hdf_file:
            self.out = self.hdf_file['output']

        self.inpbuf = None
        self.outbuf = None
        self.allocated = False

    def allocate(self):
        """
        After the input and output (`self.inp` and `self.out)` have been
        set this function will allocate the host and device buffers
        for the mini-batches.

        The host buffer is referenced as `self.mini_batch_in` and `self.mini_batch_out`, and
        stored on device as `self.inbuf` and `self.outbuf`.
        """
        if not self.allocated:
            self.allocate_inputs()
            self.allocate_outputs()
            self.allocated = True

    def allocate_inputs(self):
        """
        Allocates the host and device input data buffers
        and any other associated storage.

        `self.inpbuf` is the on-device buffer for the input minibatch
        `self.mini_batch_in` is the on-host buffer for the input minibatch
        `self.mean` is the on-device buffer of the mean array
        """
        # on device minibatch_buffer (input)
        self.inpbuf = self.be.iobuf(self.inp.shape[1])

        # setup host buffer for a mini_batch
        self.mini_batch_in = np.zeros(self.inpbuf.shape)

        self.mean = None
        # the 'mean' dataset is the the mean values to subtract
        if 'mean' in self.hdf_file:
            mns_ = np.array(self.hdf_file['mean']).flatten()
            if mns_.size != self.inp.shape[1]:
                # channel by channel mean
                # there should be 1 element per channel
                assert mns_.size == self.lshape[0], 'mean image size mismatch'
                # need to have 2-d array for broadcasting
                mns_ = mns_.reshape((self.lshape[0], 1)).copy()
                # make channel-by-channel mean subtraction view
                self.meansub_view = self.inpbuf.reshape((self.lshape[0], -1))
            else:
                self.meansub_view = self.inpbuf

            self.mean = self.be.array(mns_)

    def allocate_outputs(self):
        """
        Allocates the host and device output data buffers
        and any other associated storage.

        `self.outbuf` is the on-device buffer for the output minibatch
        `self.mini_batch_out` is the on-host buffer for the output minibatch
        """
        self.outbuf = None
        if 'output' in self.hdf_file:
            self.outbuf = self.be.iobuf(self.out.shape[1])
            self.mini_batch_out = np.zeros(self.outbuf.shape)

    def gen_input(self, mini_batch):
        """
        Function to handle any preprocessing before pushing an input
        mini-batch to the device.  For example, mean subtraction etc.

        Arguments:
            mini_batch (ndarray): M-by-N array where M is the flatten
                                  input vector size and N is the batch size
        """
        self.inpbuf[:] = mini_batch
        # mean subtract
        if self.mean is not None:
            self.meansub_view[:] = -self.mean + self.meansub_view

    def gen_output(self, mini_batch):
        """
        Function to handle any preprocessing before pushing an output
        mini-batch to the device.  For example, one-hot generation.

        Arguments:
            mini_batch (ndarray): M-by-N array where M is the flatten
                                  output vector size and N is the batch size
        """
        self.outbuf[:] = mini_batch

    def __del__(self):
        self.cleanup()
        # needed for python3 to exit cleanly
        super(HDF5Iterator, self).__del__()

    def cleanup(self):
        """
        Closes the HDF file.
        """
        self.hdf_file.close()

    def reset(self):
        """
        Resets the index to zero.
        """
        self.start = 0

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch. A minibatch includes both features and
            labels.
        """
        if not self.allocated:
            self.allocate()
        full_shape = list(self.lshape)
        full_shape.append(-1)

        mini_batch_in = self.mini_batch_in
        if self.outbuf is not None:
            mini_batch_out = self.mini_batch_out
        for i1 in range(self.start, self.ndata, self.be.bsz):
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            if i2 == self.ndata:
                self.start = self.be.bsz - bsz

            # load mini batch on host
            xdev = self.inp
            mini_batch_in[:, :bsz] = xdev[i1:i2, :].T.astype(np.float32)
            if self.be.bsz > bsz:
                mini_batch_in[:, bsz:] = xdev[:(self.be.bsz - bsz), :].T.astype(np.float32)

            # push to device
            self.gen_input(mini_batch_in)

            if self.outbuf is not None:
                mini_batch_out[:, :bsz] = self.out[i1:i2].T
                if self.be.bsz > bsz:
                    mini_batch_out[:, bsz:] = self.out[:(self.be.bsz - bsz)].T

                self.gen_output(mini_batch_out)

            inputs = self.inpbuf
            targets = self.outbuf
            yield (inputs, targets)
