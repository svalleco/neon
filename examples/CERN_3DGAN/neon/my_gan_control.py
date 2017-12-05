import numpy as np
import time
import os, errno
from shutil import copyfile, copy2

'''test dropout .75, 256 affine layer before top layer the discriminator'''
# control parameters of my_gan
#debugging an printing
my_gan_control_debug = True
my_gan_control_plot_matrix = True
my_gan_control_print_tensor_examples = False
my_gan_control_compute_all_costs = True
my_gan_control_save_training_progress = False #hdf files
my_gan_control_save_prm = False #prm files
my_gan_control_feed_dummy_data = "no" #ones, noise
my_gan_control_print_image_of_training_on_data = False

#data mng
my_gan_control_use_hdf5_iterator = True
my_gan_control_data_saving_freq = 60 #this must be a multiple of my_gan_control_k


#Initializations
my_gan_control_my_xavier_discr = False # with True will lead to NANs in discriminator fake/real output. Why?
my_gan_control_my_xavier_gen = False
my_gan_control_gaussian_scale_init_for_generator = 0.001
my_gan_control_gaussian_scale_init_for_generator_top_layer = my_gan_control_gaussian_scale_init_for_generator * 0.01
my_gan_control_gaussian_scale_init_for_discriminator = 0.1

#duration and batchsize, latent size
my_gan_control_batch_size = 64
my_gan_control_nb_epochs = 50 # generator trainings
my_gan_control_latent_size = 1024 # should I increase this?

#optimizer and cost function
my_gan_control_LR_generator = 1e-4 #not used for RMSProp   should I reduce these LRs?
my_gan_control_LR_discriminator = 1e-4 #not used for RMSProp
my_gan_control_relative_vs_meansquared = "MeanSquared" #MeanSquared vs RelativeCost
my_gan_control_generator_optimizer = "Adam" # Adam; RMSProp; anything else it will set to GradientDescent
my_gan_control_discriminator_optimizer = "Adam" # Adam; RMSProp; anything else it will set to GradientDescent
my_gan_control_cost_function = "Modified" #  Wasserstein, Modified, Original
# with Wasserstein on cost displayed is weird and bouncing from negative to positive; review gradient clipping;
# check why it is so small; learning happen however.
# TODO indeed: also wgan_param_clamp must be enabled by this set to Wasserstein
'''
model, cost = create_model(dis_model=args.dmodel, gen_model=args.gmodel,
                           cost_type='wasserstein', noise_type='normal',
                           im_size=32, n_chan=1, n_noise=128,
                           n_gen_ftr=args.n_gen_ftr, n_dis_ftr=args.n_dis_ftr,
                           depth=4, n_extra_layers=4,
                           batch_norm=True, dis_iters=5,
                           wgan_param_clamp=0.01, wgan_train_sched=True
'''
my_gan_control_train_schedule = False #
my_gan_control_param_clamp = None #None, 1.0, 0.01


#model configuration
my_gan_control_alpha = (.92, 0.02, 0.02) # : R/F, SUMEcal, Ep
my_gan_control_lshape = (1, 25, 25, 25)
my_gan_control_discriminator_option = 1  # 1 or 2 ; 3 = all convolution
my_gan_control_generator_option = 2 # 1 or 2, 3 = all deconvolution
my_gan_control_data_normalization = "no"# "for_tanh_output" "for_logistic_output"; else or nothing for relu (as output layer of generator)
my_gan_control_gen_top = "tanh" #"tanh", "logistic", anything else will be lrelu defined in the generator definition todo: in energy_dataset review the mean computation!
my_gan_control_k = 3 #>0 should I increase this?
my_gan_control_gen_times = 1 #2 it was as in Keras
my_gan_contol_train_gen = True
my_gan_control_trick = True #enabling backprop on generator making discriminator think that is data: trick as in Keras implementation??
my_gan_control_inference_only = False #CHANGE IT accordingly!
my_gan_control_drop_out_rate = 0.9

# settings for inference only
if my_gan_control_inference_only:
    my_gan_control_inference_dir = "inference_results/"
else:
    # settings for training
    my_gan_control_run_random_prefix = str(int(np.random.randint(1,10000000, 1))) + "_"
    print("################## SIMULATION PREFIX FOR OUTPUT IDENTIFICATION IS: {} ####################".format(my_gan_control_run_random_prefix))
    my_gan_control_timestamp = time.strftime("%d-%m-%Y-%H-%M")
    my_gan_results_dir = "results_{}_{}/".format(my_gan_control_timestamp, my_gan_control_run_random_prefix)
    try:
        os.makedirs(my_gan_results_dir)
        print("Created directory {}".format(my_gan_results_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #copy this file into results dir as to track how they were obtained
    my_gan_control_this_file_name = "my_gan_control.py" #os.path.basename(__file__)
    my_gan_control_destination_file_name = my_gan_results_dir + my_gan_control_run_random_prefix +  my_gan_control_this_file_name
    copyfile( my_gan_control_this_file_name,  my_gan_control_destination_file_name)
    my_gan_control_layer_file_name = "my_gan_layers.py"
    my_gan_control_destination_file_name = my_gan_results_dir + my_gan_control_run_random_prefix +  my_gan_control_layer_file_name
    copyfile( my_gan_control_layer_file_name,  my_gan_control_destination_file_name )



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
