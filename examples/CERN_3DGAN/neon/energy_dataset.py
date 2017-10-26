import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neon.data import NervanaDataIterator

def temp_3Ddata(fileName):
   f = h5py.File(fileName,"r")
   data = f.get('ECAL')
   dtag =f.get('target')
   mykeys = f.keys()
   xtr = np.array(data)
   print xtr.shape
   xtr =np.array(np.expand_dims(xtr, axis=1))
   print xtr.shape
   plt.imshow(xtr[1,0, :, :, 12])
   #plt.plot(X[0, 12, :, :])
   plt.savefig('test_xy')
   plt.imshow(xtr[1, 0, :, 12, :])
   #plt.plot(X[0, 12, :, :])
   plt.savefig('test_xz')
   aa = np.reshape(xtr, (xtr.shape[0], 25*25*25)) 
   sumE = np.sum(aa, axis=(1))
   Epart = np.array(dtag)
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
        self.shape = lshape
        self.start = 0
        self.ndata = self.X.shape[0]
        self.nfeatures =self.X.shape[1] #Nchannels*W*H*D
        self.nyfeatures =self.Y.shape[1] #Nchannels*W*H*D
        self.nbatches = int(self.ndata/self.be.bsz)
        self.dev_X = self.be.zeros((self.nfeatures, self.be.bsz))
        self.dev_Y = self.be.zeros((self.nyfeatures, self.be.bsz))  # 2 targets: primaryE, sumEcal

    def reset(self):
        self.start = 0

    def __iter__(self):
        # 3. loop through minibatches in the dataset
        for index in range(self.start, self.ndata - self.be.bsz, self.be.bsz):
            # 3a. grab the right slice from the numpy arrays
            inputs = self.X[index:(index+self.be.bsz),:]
            targets = self.Y[index:(index+self.be.bsz),:]
            
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
            # the second should of shape (4, batch_size)
            yield (self.dev_X, self.dev_Y)
