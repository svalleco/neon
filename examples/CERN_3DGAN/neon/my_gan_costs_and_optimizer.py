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
from my_gan_model import myGAN, myGenerativeAdversarial
from my_gan_layers import discriminator, generator
import numpy as np
from sklearn.model_selection import train_test_split

from neon.transforms.cost import Cost
from neon.optimizers.optimizer import Optimizer



class RelativeCost(Cost):

    """
    Average Relative Absolute Error cost function. Computes :math:`\\frac{1}{N}\\abs (sum_i (y_i-t_i)/t_i)`.
    """

    def __init__(self):
        """
        Define the cost function and its gradient as lambda functions.
        """
        eps = 10e-8 # to prevent inf
        self.func = lambda y, t: self.be.mean(self.be.absolute(self.be.divide((y - t), self.be.absolute(t) + eps)), axis=0)
        self.funcgrad = lambda y, t: self.be.multiply(self.be.multiply(self.be.absolute(self.be.reciprocal(t + eps)), y.shape[0]), self.be.sgn(y-t))# 1/N *1/|t| * sgn(y-t): sgn: ---->(y-t)/|y-t|


class DummyOptimizer(Optimizer):

    def optimize(self, layer_list, epoch):
        print("CALLED DUMMY OPT")
        pass