from my_gan_control import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neon.data import NervanaDataIterator

import os
import h5py
import logging
import numpy as np
from neon.data import HDF5Iterator


from neon.data import ArrayIterator
logger = logging.getLogger(__name__)


def temp_3Ddata(fileName):
   f = h5py.File(fileName,"r")
   data = f.get('ECAL')
   dtag =f.get('target')
   ECAL_attrs = data.attrs.keys()
   target_attrs = dtag.attrs.keys()
   mykeys = f.keys()
   xtr = np.array(data) # N x X x Y x Z

   if my_debug:
       xtr = xtr[:10000,:,:,:] #just to speed up debugging

   print ("xtr shape is {}".format(xtr.shape))
   xtr =np.array(np.expand_dims(xtr, axis=1))
   print xtr.shape

   #plotting
   plt.figure()
   if plot_matrix:
       plt.imshow(xtr[0, 0, :, :, 12])
   else:
       plt.plot(xtr[0, 0, :, :, 12])
   plt.savefig('results/test_xy')
   plt.close()

   #plt.imshow(xtr[1, 0, :, 12, :])
   plt.figure()
   if plot_matrix:
       plt.imshow(xtr[0, 0, :, 12, :])
   else:
       plt.plot(xtr[0, 0, :, 12, :])
   plt.savefig('results/test_xz')
   plt.close()

   plt.figure()
   if plot_matrix:
       plt.imshow(xtr[0, 0, 12, :, :])
   else:
       plt.plot(xtr[0, 0, 12, :, :])
   plt.savefig('results/test_yz')
   plt.close()


   # N x W*H*D
   aa = np.reshape(xtr, (xtr.shape[0], 25*25*25))
   sumE = np.sum(aa, axis=(1))
   Epart = np.array(dtag)

   if my_debug:
       Epart = Epart[:10000,:] #just to speed up debugging

   labels = np.stack((Epart[:, 1]/100, sumE), axis=1)
   return aa.astype(np.float32), labels.astype(np.float32)




def get_output(outputFileName):
   f = h5py.File(outputFileName,"r")
   data = f.get("ECAL")
   x = np.array(data)
   return x.astype(np.float32)


class EnergyData(NervanaDataIterator):
    def __init__(self, X, Y, lshape):
        self.X = X 
        self.Y = Y
        self.shape, self.lshape = lshape, lshape
        self.start = 0
        self.ndata = self.X.shape[0]
        self.nfeatures =self.X.shape[1] #Nchannels*W*H*D
        self.nyfeatures =self.Y.shape[1]
        self.nbatches = int(self.ndata/self.be.bsz)
        self.dev_X = self.be.zeros((self.nfeatures, self.be.bsz))
        self.dev_Y = self.be.zeros((self.nyfeatures, self.be.bsz))  # 2 targets: primaryE, sumEcal

    def reset(self):
        self.start = 0

    def __iter__(self):
        # 3. loop through minibatches in the dataset
        for index in range(self.start, self.ndata - self.be.bsz, self.be.bsz):
            bsz = self.be.bsz
            # 3a. grab the right slice from the numpy arrays
            inputs = self.X[index:(index + bsz),:]
            targets = self.Y[index:(index + bsz),:]
            
            # The arrays X and Y data are in shape (batch_size, num_features),
            # but the iterator needs to return data with shape (num_features, batch_size).
            # here we transpose the data, and then store it as a contiguous array. 
            # numpy arrays need to be contiguous before being loaded onto the GPU.
            inputs = np.ascontiguousarray(inputs.T)
            targets = np.ascontiguousarray(targets.T)

            # here we test your implementation
            # your slice has to have the same shape as the GPU tensors you allocated
            assert inputs.shape == self.dev_X.shape, \
                   "inputs has shape {}, but dev_X is {}".format(inputs.shape, self.dev_X.shape)
            assert targets.shape == self.dev_Y.shape, \
                   "targets has shape {}, but dev_Y is {}".format(targets.shape, self.dev_Y.shape)
            
            # 3b. transfer from numpy arrays to device
            # - use the GPU memory buffers allocated previously,
            #    and call the myTensorBuffer.set() function. 
            self.dev_X.set(inputs)
            self.dev_Y.set(targets)
            
            # 3c. yield a tuple of the device tensors.
            # the first should of shape (num_features, batch_size)
            # the second should of shape (2, batch_size)
            yield (self.dev_X, self.dev_Y)




class my_gan_HDF5Iterator(ArrayIterator):
    """
    Data iterator which uses an HDF5 file as the source of the data, useful when
    the entire HDF5 dataset cannot fit into memory (for smaller datasets, use the ArrayIterator).

    To initialize the HDF5Iterator, simply call::

        train_set = HDF5Iterator('your_data_path.h5')

    The HDF5 file format must contain the following datasets:

    - `ECAL` (ndarray):
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
    - `target` (ndarray):
                        An optional dataset which, if supplied, will be
                        used at the target/expected output of the network. the
                        array should have the shape `(N, M)` where `N` is the number
                        of items (must match the `N` dim of the input set)
                        and `M` is the size of the output data which should match
                        size of output from the output layer of the network.

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
        self.inp = self.hdf_file['ECAL']
        self.ndata = self.inp.shape[0]

        # must have at least 1 minibatch of data in the file
        assert self.ndata >= self.be.bsz
        self.start = 0

        # the input array unflattened size
        self.lshape = (1, 25, 25, 25)# we know the lshape; original code was: tuple(self.inp.attrs['lshape'])
        self.shape = self.lshape

        if 'target' in self.hdf_file:
            self.out = self.hdf_file['target']

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
        pr = self.inp.shape[1:]
        pr = np.array(pr)
        pr = pr.prod()
        self.inpbuf = self.be.iobuf(pr)

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
        if 'target' in self.hdf_file:
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
        for i1 in range(self.start, self.ndata - self.be.bsz, self.be.bsz): # otherwise at the end of the epoch i got issues
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            if i2 == self.ndata:
                self.start = self.be.bsz - bsz

            # load mini batch on host
            xdev = self.inp
            # Here it is where we transpose from hdf5 format (our input is N x 25 x25 x25) to Neon format, 25*25*25 x N;
            my_buf = np.array(xdev[i1:i2, :]).astype(np.float32) # converting batch to numpy
            my_buf = np.moveaxis(my_buf, 0, -1) # moving batch axis at the end, keeping the x,y,z order (now X,Y,Z,N)
            mini_batch_in[:, :bsz] = my_buf.reshape((np.array(self.lshape)).prod(), bsz) #reshaping into 25*25*25xN
            # transposing above does N,x,y,z into z,y,x,N



            if self.be.bsz > bsz:
                mini_batch_in[:, bsz:] = xdev[:(self.be.bsz - bsz), :].T.astype(np.float32)

            # push to device
            self.gen_input(mini_batch_in)

            if self.outbuf is not None:
                mini_batch_out[:, :bsz] = self.out[i1:i2].T

                #setting label values from hdf5 data file
                # here mini_batch_out should have [0,:] = +/-11 for particle identification and [1,:] Ep
                # we want to have Ep in [0,:] and SumE in [1,:]
                sumE = np.sum(mini_batch_in, axis=(0))
                mini_batch_out[0, :] = mini_batch_out[1, :] / 100.
                mini_batch_out[1, :] = sumE

                if self.be.bsz > bsz:
                    mini_batch_out[:, bsz:] = self.out[:(self.be.bsz - bsz)].T

                self.gen_output(mini_batch_out) #Energy rescaling as in temp3D_data above_

            inputs = self.inpbuf
            targets = self.outbuf
            yield (inputs, targets)


'''
  File "/home/azanetti/CERN/neon/examples/CERN_3DGAN/neon/energy_gan.py", line 154, in <module>
    main()
  File "/home/azanetti/CERN/neon/examples/CERN_3DGAN/neon/energy_gan.py", line 136, in main
    cost=cost, callbacks=callbacks)
  File "/home/azanetti/CERN/neon/neon/models/model.py", line 183, in fit
    self._epoch_fit(dataset, callbacks)
  File "/home/azanetti/CERN/neon/examples/CERN_3DGAN/neon/my_gan_model.py", line 262, in _epoch_fit
    for mb_idx, (x, labels) in enumerate(dataset):
  File "/home/azanetti/CERN/neon/examples/CERN_3DGAN/neon/energy_dataset.py", line 339, in __iter__
    mini_batch_in[:, bsz:] = xdev[:(self.be.bsz - bsz), :].T.astype(np.float32)
ValueError: could not broadcast input array from shape (25,25,25,64) into shape (15625,64)
Exception TypeError: 'super(type, obj): obj must be an instance or subtype of type' in <bound method my_gan_HDF5Iterator.__del__ of <energy_dataset.my_gan_HDF5Iterator object at 0x46fc110>> ignored
Exception TypeError: 'super(type, obj): obj must be an instance or subtype of type' in <bound method my_gan_HDF5Iterator.__del__ of <energy_dataset.my_gan_HDF5Iterator object at 0x46eaf90>> ignored
'''
