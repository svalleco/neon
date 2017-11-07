from neon.initializers import Gaussian, Constant, Xavier
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm, BranchNode, GeneralizedCost
from neon.layers.layer import Linear, Reshape
from neon.layers.container import Tree, Multicost, LayerContainer, GenerativeAdversarial
from neon.transforms import Rectlin, Logistic, GANCost, Tanh, MeanSquared, Identity
from neon.layers.layer import Dropout
from my_gan_control import *


def discriminator():
    # setup weight initialization function

    if discriminator_option_1:

        if my_xavier_discr:
            init =  Xavier()
        else:
            init = Gaussian(scale=my_gaussian_scale_init_for_discriminator)

        if my_control_cost_function == "Wasserstein":
            Top_Layer = Linear(nout=1, init=init)
        else:
            Top_Layer = Affine(nout=1, init=init, bias=init, activation=Logistic())

        # discriminator using convolution layers
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
                    Top_Layer #   Affine(nout=1, init=init, bias=init, activation=Logistic()) # for non-Wasserstein Identity() per Wasserstein?
                    ] #real/fake
        branch2 = [b2,
                   Affine(nout=1, init=init, bias=init, activation=Identity())] #E primary
        branch3 = [b1,
                   Linear(nout=1, init=Constant(val=1.0), name="NotOptimizeLinear")] #SUM ECAL
    else:

        #setup weight initialization function
        if my_xavier_discr:
            init = Xavier()
        else:
            init = Gaussian(scale=my_gaussian_scale_init_for_discriminator)

        if my_control_cost_function == "Wasserstein":
            Top_Layer = Linear(nout=1, init=init)
        else:
            Top_Layer = Affine(nout=1, init=init, bias=init, activation=Logistic())

        # discriminator using convolution layers
        lrelu = Rectlin(slope=0.1)  # leaky relu for discriminator
        # sigmoid = Logistic() # sigmoid activation function
        conv1 = dict(init=init, batch_norm=False, activation=lrelu, bias=init)
        conv2 = dict(init=init, batch_norm=False, activation=lrelu, padding=2, bias=init)
        conv3 = dict(init=init, batch_norm=False, activation=lrelu, padding=1, bias=init)
        b1 = BranchNode("b1")
        b2 = BranchNode("b2")
        branch1 = [b1,
                   Conv((5, 5, 5, 32), **conv1),
                   Dropout(keep=0.8),
                   Conv((5, 5, 5, 8), **conv2),
                   BatchNorm(),
                   Dropout(keep=0.8),
                   Conv((5, 5, 5, 8), **conv2),
                   BatchNorm(),
                   Dropout(keep=0.8),
                   Conv((5, 5, 5, 8), **conv3),
                   BatchNorm(),
                   Dropout(keep=0.8),
                   Pooling((2, 2, 2)),
                   # Affine(1024, init=init, activation=lrelu),
                   # BatchNorm(),
                   # Affine(1024, init=init, activation=lrelu),
                   # BatchNorm(),
                   b2,
                   Top_Layer #Affine(nout=1, init=init, bias=init, activation=Logistic())
                   # for non-Wasserstein Identity()/Linear() per Wasserstein
                   ]  # real/fake
        branch2 = [b2,
                   Affine(nout=1, init=init, bias=init, activation=Identity())]  # E primary
        branch3 = [b1,
                   Linear(nout=1, init=Constant(val=1.0), name="NotOptimizeLinear")]  # SUM ECAL

    if my_three_lines:
        D_layers = Tree([branch1, branch2, branch3], name="Discriminator", alphas=my_alpha)
        print("Using Three lines with alpha = {}".format(my_alpha))
    else:
        D_layers = Tree([branch1, branch2, branch3], name="Discriminator", alphas=my_alpha_balanced)
    return D_layers


def generator():

    if generator_option_1:
        if my_xavier_gen:
            init_gen = Xavier()
        else:
            init_gen = Gaussian(scale=my_gaussian_scale_init_for_generator)
        lrelu = Rectlin(slope=0.1)  # leaky relu
        relu = Rectlin(slope=0)  # relu for generator
        pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
        str1 = dict(str_h=2, str_w=2, str_d=2)
        conv1 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad1, strides=str1, bias=init_gen)
        pad2 = dict(pad_h=2, pad_w=2, pad_d=2)
        str2 = dict(str_h=2, str_w=2, str_d=2)
        conv2 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad2, strides=str2, bias=init_gen)
        pad3 = dict(pad_h=0, pad_w=0, pad_d=0)
        str3 = dict(str_h=1, str_w=1, str_d=1)
        conv3 = dict(init=init_gen, batch_norm=False, activation=Tanh(), padding=pad3, strides=str3, bias=init_gen) # Rectlin()
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
    else:
        lrelu = Rectlin(slope=0.1)  # leaky relu
        if my_xavier_gen:
            init_gen = Xavier()
        else:
            init_gen = Gaussian(scale=my_gaussian_scale_init_for_generator)
        relu = Rectlin(slope=0)  # relu for generator
        # pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
        pad1 = dict(pad_h=0, pad_w=0, pad_d=0)
        # str1 = dict(str_h=2, str_w=2, str_d=2)
        str1 = dict(str_h=1, str_w=1, str_d=1)
        conv1 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad1, strides=str1, bias=init_gen)
        # pad2 = dict(pad_h=2, pad_w=2, pad_d=2
        # str2 = dict(str_h=2, str_w=2, str_d=2)
        pad2 = dict(pad_h=0, pad_w=1, pad_d=0)
        str2 = dict(str_h=2, str_w=2, str_d=2)
        conv2 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad2, strides=str2, bias=init_gen)
        pad3 = dict(pad_h=0, pad_w=0, pad_d=0)
        str3 = dict(str_h=1, str_w=1, str_d=1)
        conv3 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad3, strides=str3, bias=init_gen)
        pad4 = dict(pad_h=0, pad_w=0, pad_d=0)
        str4 = dict(str_h=1, str_w=1, str_d=1)
        conv4 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad4, strides=str4, bias=init_gen)
        conv5 = dict(init=init_gen, batch_norm=False, activation=Tanh(), padding=pad4, strides=str4, bias=init_gen) # Rectlin()
        bg = BranchNode("bg")
        branchg = [bg,
                   # Affine(1024, init=init_gen, bias=init_gen, activation=relu),
                   # BatchNorm(),
                   Affine(8 * 8 * 7 * 7, init=init_gen, bias=init_gen),
                   Reshape((8, 7, 7, 8)),
                   Deconv((6, 6, 8, 64), **conv1),  # 14x14x14
                   BatchNorm(),
                   # Linear(5 * 14 * 14 * 14, init=init),
                   # Reshape((5, 14, 14, 14)),
                   Deconv((6, 6, 8, 6), **conv2),  # 27x27x27
                   BatchNorm(),
                   Deconv((2, 2, 6, 6), **conv3),  # 27x27x27
                   Conv((4, 4, 6, 8), **conv4),
                   Conv((2, 2, 10, 1), **conv5)
                   ]

    G_layers = Tree([branchg], name="Generator")
    return G_layers



