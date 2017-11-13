from neon.initializers import Gaussian, Constant, Xavier
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm, BranchNode, GeneralizedCost
from neon.layers.layer import Linear, Reshape
from neon.layers.container import Tree, Multicost, LayerContainer, GenerativeAdversarial
from neon.transforms import Rectlin, Logistic, GANCost, Tanh, MeanSquared, Identity
from neon.layers.layer import Dropout
from my_gan_control import *


def discriminator():
    # setup weight initialization function


    if my_xavier_discr:
        init = Xavier()
    else:
        init = Gaussian(scale=my_gaussian_scale_init_for_discriminator)

    if my_control_cost_function == "Wasserstein":
        Top_Layer = Linear(nout=1, init=init)
    else:
        Top_Layer = Affine(nout=1, init=init, bias=init, activation=Logistic())

    if discriminator_option == 1:
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
                   Affine(nout=1, init=init, bias=init, activation=Rectlin())] #E primary
        branch3 = [b1,
                   Linear(nout=1, init=Constant(val=1.0), name="NotOptimizeLinear")] #SUM ECAL
    else: #discriminator_option == 2
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
    if my_xavier_gen:
        init_gen = Xavier()
    else:
        init_gen = Gaussian(scale=my_gaussian_scale_init_for_generator)

    if generator_option == 1:
        lrelu = Rectlin(slope=0.1)  # leaky relu
        relu = Rectlin(slope=0)  # relu for generator
        pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
        str1 = dict(str_h=2, str_w=2, str_d=2)
        conv1 = dict(init=init_gen, batch_norm=False, activation=relu, padding=pad1, strides=str1, bias=init_gen)
        pad2 = dict(pad_h=2, pad_w=2, pad_d=2)
        str2 = dict(str_h=2, str_w=2, str_d=2)
        conv2 = dict(init=init_gen, batch_norm=False, activation=relu, padding=pad2, strides=str2, bias=init_gen)
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
    elif generator_option == 2:
        relu = Rectlin(slope=0)  # relu for generator
        # pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
        pad1 = dict(pad_h=0, pad_w=0, pad_d=0)
        # str1 = dict(str_h=2, str_w=2, str_d=2)
        str1 = dict(str_h=1, str_w=1, str_d=1)
        conv1 = dict(init=init_gen, batch_norm=False, activation=relu, padding=pad1, strides=str1, bias=init_gen)
        # pad2 = dict(pad_h=2, pad_w=2, pad_d=2
        # str2 = dict(str_h=2, str_w=2, str_d=2)
        pad2 = dict(pad_h=0, pad_w=1, pad_d=0)
        str2 = dict(str_h=2, str_w=2, str_d=2)
        conv2 = dict(init=init_gen, batch_norm=False, activation=relu, padding=pad2, strides=str2, bias=init_gen)
        pad3 = dict(pad_h=0, pad_w=0, pad_d=0)
        str3 = dict(str_h=1, str_w=1, str_d=1)
        conv3 = dict(init=init_gen, batch_norm=False, activation=relu, padding=pad3, strides=str3, bias=init_gen)
        pad4 = dict(pad_h=0, pad_w=0, pad_d=0)
        str4 = dict(str_h=1, str_w=1, str_d=1)
        conv4 = dict(init=init_gen, batch_norm=False, activation=relu, padding=pad4, strides=str4, bias=init_gen)
        conv5 = dict(init=init_gen, batch_norm=False, activation=Tanh(), padding=pad4, strides=str4, bias=init_gen) # Rectlin()/Tanh
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
    else: #generator_option == 3
        # generator using "decovolution" layers
        pad_hwd_111 = dict(pad_h=1, pad_w=1, pad_d=1)
        str_hwd_222 = dict(str_h=2, str_w=2, str_d=2)
        relu = Rectlin(slope=0)  # relu for generator
        conv = dict(init=init_gen, batch_norm=True, activation=relu)
        convp1_a = dict(init=init_gen, batch_norm=True, activation=relu, padding=pad_hwd_111)
        convp1s2_a = dict(init=init_gen, batch_norm=True, activation=relu, padding=pad_hwd_111, strides=str_hwd_222)
        bg = BranchNode("bg")
        branchg = [bg,
                   Affine(7 * 7 * 7 * 1, init=init_gen, bias=init_gen),
                   Reshape((1, 7, 7, 7)),
                   Deconv((1, 1, 1, 16), name="G11", **conv),  # inshape: 4,7,7,7 outshape: 16,7,7,7 V
                   Deconv((3, 3, 3, 64), name="G12", **convp1_a),  # outshape: 64,7,7,7 V - padding in d was 0 by def?
                   Deconv((3, 3, 3, 96), name="G21", **convp1s2_a),  # outshape: 96,13,13,13 V
                   Deconv((3, 3, 3, 96), name="G22", **convp1_a),  # outshape: 96,13,13,13
                   Deconv((3, 3, 3, 64), name="G31", **convp1s2_a),  # outshape: 64,25,25,25
                   Deconv((3, 3, 3, 16), name="G32", **convp1_a), # outshape: 16,25,25,25
                   Deconv((3, 3, 3, 1), name="G_out",  # outshape: 1,25,25,25
                          init=init_gen, batch_norm=True, padding=pad_hwd_111,
                          activation=Tanh())] #Logistic(shortcut=False))]

    G_layers = Tree([branchg], name="Generator_andrea")
    return G_layers


def discriminator_andrea():
    #Todo:
    # setup weight initialization function
    init = Gaussian(scale=0.05)
    # discriminiator using convolution layers
    lrelu = Rectlin(slope=0.1)  # leaky relu for discriminator
    conv = dict(init=init, batch_norm=True, activation=lrelu)
    convp1 = dict(init=init, batch_norm=True, activation=lrelu, padding=1)
    convp1s2 = dict(init=init, batch_norm=True, activation=lrelu, padding=1, strides=2)
    D_layers = [Conv((3, 3, 96), name="D11", **convp1),
                Conv((3, 3, 96), name="D12", **convp1s2),
                Conv((3, 3, 192), name="D21", **convp1),
                Conv((3, 3, 192), name="D22", **convp1s2),
                Conv((3, 3, 192), name="D31", **convp1),
                Conv((1, 1, 16), name="D32", **conv),
                Conv((7, 7, 1), name="D_out",
                     init=init, batch_norm=False,
                     activation=Logistic(shortcut=False))]