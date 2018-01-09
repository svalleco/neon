import numpy as np
import time
import os, errno
from shutil import copyfile, copy2

# control parameters of my_gan

#remote server
mgc_local_gdansk = False
#debugging an printing
mgc_debug = True
mgc_plot_matrix = True
mgc_print_tensor_examples = False
mgc_compute_all_costs = True
mgc_save_training_progress = False  # hdf files
mgc_save_prm = False  # prm files
mgc_feed_dummy_data = "no"  # ones, noise
mgc_print_image_of_training_on_data = False

#data mng
mgc_use_hdf5_iterator = True
mgc_data_saving_freq = 100 #this must be a multiple of mgc_k


#Initializations
mgc_my_xavier_discr = False  # with True for dicriminator leads to NANs in discriminator fake/real output. Why?
mgc_my_xavier_gen = False
mgc_gaussian_scale_init_for_generator = 0.001
mgc_gaussian_scale_init_for_generator_top_layer = mgc_gaussian_scale_init_for_generator * 0.01
mgc_gaussian_scale_init_for_discriminator = 0.05

#duration and batchsize, latent size
mgc_batch_size = 128
mgc_nb_epochs = 70  # generator trainings
mgc_latent_size = 1024  # should I increase this?

#optimizer and cost function
mgc_LR_generator = 5e-5 #
mgc_LR_discriminator = 5e-5 #
mgc_relative_vs_meansquared = "RelativeCost"  # MeanSquared vs RelativeCost
mgc_generator_optimizer = "RMSProp"  # Adam; RMSProp; anything else it will set to GradientDescent
mgc_discriminator_optimizer = "RMSProp"  # Adam; RMSProp; anything else it will set to GradientDescent
mgc_cost_function = "Original"  # Wasserstein, Modified, Original
# with Wasserstein on cost displayed is weird and bouncing from negative to positive; review gradient clipping;
# check why it is so small; learning happen however.
# TODO indeed: also wgan_param_clamp must be enabled by this set to Wasserstein
mgc_train_schedule = False  # This causes Generator to be trained every 100 iterations for the Discriminator and k seems to be ignored if this is true
mgc_param_clamp = None  # None, 1.0, 0.01

#model configuration
mgc_alpha = (1.0, 0.02, 0.05) # : R/F, SUMEcal, Ep
mgc_lshape = (1, 25, 25, 25)
mgc_discriminator_option = 1  # 1 or 2 ; 3 = all convolution
mgc_generator_option = 2  # 1 or 2, 3 = all deconvolution
mgc_data_normalization = "no" #  "for_tanh_output" "for_logistic_output"; else or nothing for relu (as output layer of generator)
mgc_gen_top = "lrelu" #"tanh", "logistic", "lrelu" anything else will be Relu defined in the generator definition todo: in energy_dataset review the mean computation!
mgc_k = 5  # >0 should I increase this?
mgc_gen_times = 1  # 2 it was as in Keras
mgc_train_gen = True
mgc_trick = True  # enabling backprop on generator making discriminator think that is data: trick as in Keras implementation??
mgc_inference_only = False  # CHANGE IT accordingly!
mgc_drop_out_rate = 0.8

# settings for inference only
if mgc_inference_only:
    mgc_inference_dir = "inference_results/"
else:
    # settings for training
    mgc_run_random_prefix = str(int(np.random.randint(1,10000000, 1))) + "_"
    print("################## SIMULATION PREFIX FOR OUTPUT IDENTIFICATION IS: {} ####################".format(mgc_run_random_prefix))
    mgc_timestamp = time.strftime("%d-%m-%Y-%H-%M")
    my_gan_results_dir = "results_{}_{}/".format(mgc_timestamp, mgc_run_random_prefix)
    try:
        os.makedirs(my_gan_results_dir)
        print("Created directory {}".format(my_gan_results_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #copy this file into results dir as to track how they were obtained
    mgc_this_file_name = "my_gan_control.py" #os.path.basename(__file__)
    mgc_destination_file_name = my_gan_results_dir + mgc_run_random_prefix +  mgc_this_file_name
    copyfile( mgc_this_file_name,  mgc_destination_file_name)
    mgc_layer_file_name = "my_gan_layers.py"
    mgc_destination_file_name = my_gan_results_dir + mgc_run_random_prefix +  mgc_layer_file_name
    copyfile( mgc_layer_file_name,  mgc_destination_file_name )



# reminder: asintotic value for Generator cost in modified form: -ln 0.5 = + 0.693147180559945
# TODO:
# would like to plot discriminator and generator cost function evolution during training check existing call backs, and verify buffers...
#
# # Check the reason for the followking:
# END OF DISCRMINATOR TRAINING: COSTS: Real/Fake cost: 1.39133143425    Ep cost: 0.0122767258435     SUMEcal cost: 1.00333465494e-11
# Epoch 29  [Train  1562/1563 batches, 1.008951902390 cost, 1688.29s]
# INFO:neon.callbacks.callbacks:Epoch 29 complete.  Train Cost 1.392721295357.
# DEBUG:neon.util.persist:serializing object to: 7273710_energy_gan-generator-22-11-2017-16-52].prm
# DEBUG:neon.util.persist:serializing object to: 7273710_energy_gan-discriminator-22-11-2017-16-52].prm
# Exception TypeError: 'super(type, obj): obj must be an instance or subtype of type' in <bound method my_gan_HDF5Iterator.__del__ of <energy_dataset.my_gan_HDF5Iterator object at 0x56805d0>> ignored
# Exception TypeError: 'super(type, obj): obj must be an instance or subtype of type' in <bound method my_gan_HDF5Iterator.__del__ of <energy_dataset.my_gan_HDF5Iterator object at 0x5680650>> ignored


# initialization is important! especially for tha last layers and when I change between activations: around 0 tanh is 0, Logistic is 1/2!!!

#which cost is displayed here: it starts with generator cost and then accunlates and does average, but it mixes disc cost... check!!
# Epoch 0   [Train ...    1/1563 batches, 0.708468735218 cost, 3.38s]
# try with WGAN now
# my_three_lines not needed as alpha = 0 works (but watch out as in noise generation is still used)
# try with uniform generation for generator feeding
# and chech the result increasiong the dimenstionality of the noise
# and uniform with dataset max to 0-1 and sigmoid

# try to decrease the learning rate!! to increase precision in the reconstruction
# and relative cost has any effect now?
