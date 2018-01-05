from neon.initializers import Gaussian, Constant, Xavier, GlorotUniform, Kaiming
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm, BranchNode, GeneralizedCost
from neon.layers.layer import Linear, Reshape, SkipNode
from neon.layers.container import Tree, SingleOutputTree, Multicost, LayerContainer, GenerativeAdversarial
from neon.transforms import Rectlin, Logistic, GANCost, Tanh, MeanSquared, Identity
from neon.layers.layer import Dropout
from my_gan_control import *


def discriminator():
    lrelu = Rectlin(slope=0.01)  # leaky relu for discriminator
    if my_gan_control_my_xavier_discr:
        init = Xavier()
    else:
        init = Gaussian(scale=my_gan_control_gaussian_scale_init_for_discriminator)

    if my_gan_control_cost_function == "Wasserstein":
        Top_Layer = Linear(nout=1, name="Discriminator", init=init)
    else:
        Top_Layer = Affine(nout=1, name="Discriminator", init=init, bias=init, activation=Logistic(shortcut=False))

    if my_gan_control_discriminator_option == 1:
        conv1 = dict(init=init, batch_norm=False, activation=lrelu, bias=init)
        conv2 = dict(init=init, batch_norm=False, activation=lrelu, padding=2, bias=init)
        conv3 = dict(init=init, batch_norm=False, activation=lrelu, padding=1, bias=init)
        b1 = BranchNode("b1")
        b2 = BranchNode("b2")
        branch0 = [b1,
                    Conv((5, 5, 5, 32), name="Discriminator", **conv1), #21x21x21
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((5, 5, 5, 32), name="Discriminator", **conv2), #21x21x21
                    BatchNorm(),
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((5, 5, 5, 64), name="Discriminator",  **conv2), #21x21x21
                    BatchNorm(),
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((5, 5, 5, 32), name="Discriminator", **conv3), #19x19x19
                    BatchNorm(),
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Pooling((2, 2, 2), op='avg'),
                    Affine(1024, init=init, name="Discriminator", activation=lrelu),
                    BatchNorm(),
                    # Affine(1024, init=init, name="Discriminator", activation=lrelu),
                    # BatchNorm(),
                    b2,
                    Top_Layer
                    ] #real/fake
        branch1 = [b1,
                   Linear(nout=1, init=Constant(val=1.0), name="NotOptimizeLinear")] #SUM ECAL
        branch2 = [b2,
                   Affine(nout=1, init=init, bias=init, activation=lrelu)] #E primary

    elif my_gan_control_discriminator_option == 2:
        # sigmoid = Logistic() # sigmoid activation function
        conv1 = dict(init=init, batch_norm=False, activation=lrelu, bias=init)
        conv2 = dict(init=init, batch_norm=False, activation=lrelu, padding=2, bias=init)
        conv3 = dict(init=init, batch_norm=False, activation=lrelu, padding=1, bias=init)
        b1 = BranchNode("b1")
        b2 = BranchNode("b2")
        branch0 = [b1,
                   Conv((5, 5, 5, 32), name="Discriminator",  **conv1),
                   Dropout(keep=my_gan_control_drop_out_rate),
                   Conv((5, 5, 5, 8), name="Discriminator", **conv2),
                   BatchNorm(),
                   Dropout(keep=my_gan_control_drop_out_rate),
                   Conv((5, 5, 5, 8), name="Discriminator", **conv2),
                   BatchNorm(),
                   Dropout(keep=my_gan_control_drop_out_rate),
                   Conv((5, 5, 5, 8), name="Discriminator", **conv3),
                   BatchNorm(),
                   Dropout(keep=my_gan_control_drop_out_rate),
                   Pooling((2, 2, 2), op='avg'),
                   # Affine(1024, init=init, activation=lrelu),
                   # BatchNorm(),
                   # Affine(1024, init=init, activation=lrelu),
                   # BatchNorm(),
                   b2,
                   Top_Layer
                   ]  # real/fake
        branch1 = [b1,
                   Linear(nout=1, init=Constant(val=1.0), name="NotOptimizeLinear")]  # SUM ECAL
        branch2 = [b2,
                   Affine(nout=1, init=init, bias=init, name="Discriminator",  activation=lrelu)]  # E primary

    else:#discriminiator using convolution layers
        if my_gan_control_cost_function == "Wasserstein":
            Top_Layer = Conv((4, 4, 4), name="Discriminator",
                         init=init, batch_norm=False,
                         activation=Linear(nout=1, init=init))
        else:
            Top_Layer = Conv((4, 4, 4, 1), name="Discriminator",
                         init=init, batch_norm=False,
                         activation=Logistic(shortcut=False))

        pad_hwd_111 = dict(pad_h=1, pad_w=1, pad_d=1)
        str_hwd_222 = dict(str_h=2, str_w=2, str_d=2)
        convp1_l1 = dict(init=init, batch_norm=False, activation=lrelu, padding=pad_hwd_111)
        convp1 = dict(init=init, batch_norm=True, activation=lrelu, padding=pad_hwd_111)
        convp1s2 = dict(init=init, batch_norm=True, activation=lrelu, padding=pad_hwd_111, strides=str_hwd_222)
        conv = dict(init=init, batch_norm=True, activation=lrelu)
        b1 = BranchNode("b1")
        b2 = BranchNode("b2")
        branch0 = [b1,
                    Conv((3, 3, 3, 48), name="Discriminator", **convp1_l1), #outshape 25x25x25
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((3, 3, 3, 96), name="Discriminator", **convp1s2), #outshape 13x13x13
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((3, 3, 3, 192), name="Discriminator", **convp1), #outshape 13x13x13
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((3, 3, 3, 96), name="Discriminator", **convp1s2), # outshape 7x7x7
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((3, 3, 3, 96), name="Discriminator", **convp1), # outshape 7x7x7
                    Dropout(keep=my_gan_control_drop_out_rate),
                    Conv((3, 3, 3, 48), name="Discriminator", **convp1s2),  # outshape 4x4x4
                    Dropout(keep=my_gan_control_drop_out_rate),
                    b2,
                    Conv((1, 1, 1, 16), name="Discriminator", **conv), # outshape 4x4x4
                    Top_Layer
                   ] # Real/Fake
        branch1 = [b1,
                   Linear(nout=1, init=Constant(val=1.0), name="NotOptimizeLinear")]  # SUM ECAL
        branch2 = [b2,
                   Affine(nout=1, init=init, bias=init, name="Discriminator", activation=lrelu)]  # E primary

    D_layers = Tree([branch0, branch1, branch2], alphas=my_gan_control_alpha) # Real/Fake, SUM ECAL, E primary
    return D_layers


def generator():
    lrelu = Rectlin(slope=0.01)  # leaky relu
    relu = Rectlin(slope=0)  # relu for generator
    if my_gan_control_my_xavier_gen:
        init_gen = Xavier()
        init_gen_top = Xavier()
    else:
        init_gen = Gaussian(scale=my_gan_control_gaussian_scale_init_for_generator)
        init_gen_top = Gaussian(scale=(my_gan_control_gaussian_scale_init_for_generator_top_layer))

    if my_gan_control_gen_top == "tanh":
        gen_top = Tanh()
    elif my_gan_control_gen_top == "logistic":
        gen_top = Logistic(shortcut=False)
    elif my_gan_control_gen_top == "lrelu":
        gen_top = lrelu
    else:
        gen_top = Rectlin()

    if my_gan_control_generator_option == 1:
        pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
        str1 = dict(str_h=2, str_w=2, str_d=2)
        conv1 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad1, strides=str1, bias=init_gen)
        pad2 = dict(pad_h=2, pad_w=2, pad_d=2)
        str2 = dict(str_h=2, str_w=2, str_d=2)
        conv2 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad2, strides=str2, bias=init_gen)
        pad3 = dict(pad_h=0, pad_w=0, pad_d=0)
        str3 = dict(str_h=1, str_w=1, str_d=1)
        conv3 = dict(init=init_gen_top, batch_norm=False, activation=gen_top, padding=pad3, strides=str3, bias=init_gen) # Rectlin()
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

    elif my_gan_control_generator_option == 2:
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

        conv5 = dict(init=init_gen_top, batch_norm=False, activation=gen_top, padding=pad4, strides=str4, bias=init_gen) # Rectlin()/Tanh
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

    else: #my_gan_control_generator_option == 3
        pad_hwd_111 = dict(pad_h=1, pad_w=1, pad_d=1)
        str_hwd_222 = dict(str_h=2, str_w=2, str_d=2)
        relu = Rectlin(slope=0)  # relu for generator
        conv = dict(init=init_gen, batch_norm=True, activation=lrelu)
        convp1_a = dict(init=init_gen, batch_norm=True, activation=lrelu, padding=pad_hwd_111)
        convp1s2_a = dict(init=init_gen, batch_norm=True, activation=gen_top, padding=pad_hwd_111, strides=str_hwd_222)
        bg = BranchNode("bg")
        branchg = [bg,
                   Affine(7 * 7 * 7 * 1, init=init_gen, bias=init_gen),
                   Reshape((1, 7, 7, 7)),
                   Deconv((1, 1, 1, 48), name="G11", **conv),  # inshape: 4,7,7,7 outshape: 96,7,7,7 V
                   Deconv((3, 3, 3, 96), name="G12", **convp1_a),  # outshape: 96,7,7,7 V - padding in d was 0 by def?
                   Deconv((3, 3, 3, 96), name="G21", **convp1s2_a),  # outshape: 192,13,13,13 V
                   Deconv((3, 3, 3, 192), name="G22", **convp1_a),  # outshape: 192,13,13,13
                   Deconv((3, 3, 3, 96), name="G31", **convp1s2_a),  # outshape: 96,25,25,25
                   Deconv((3, 3, 3, 48), name="G32", **convp1_a), # outshape: 96,25,25,25
                   Deconv((3, 3, 3, 1), name="G_out",  # outshape: 1,25,25,25
                          init=init_gen_top, batch_norm=True, padding=pad_hwd_111,
                          activation=gen_top)]

    G_layers = Tree([branchg])
    return G_layers

