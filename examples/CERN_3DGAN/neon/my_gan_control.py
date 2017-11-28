import numpy as np
import time
import os, errno
from shutil import copyfile, copy2

''' Run description: testing effect of normalizin in 0-1. Other thing like in run 7155867 '''
# control parameters of my_gan
#debugging an printing
my_debug = True
plot_matrix = True
my_gan_control_print_tensor_examples = False
my_compute_all_costs = True
save_training_progress = False #hdf files
my_gan_control_save_prm = False #prm files
my_gan_control_feed_dummy_data = "no" #ones, noise
my_gan_control_print_image_of_training_on_data = False

#data mng
my_use_hdf5_iterator = True
data_saving_freq = 50 #this must be a multiple of my_gan_k

#Initializations
my_xavier_discr = False # with True will lead to NANs in discriminator fake/real output. Why?
my_xavier_gen = False
my_gaussian_scale_init_for_generator = 0.001
my_gaussian_scale_init_for_discriminator = 0.01

#duration and batchsize, latent size
my_gan_control_batch_size = 128
my_gan_control_nb_epochs = 30
my_gan_control_latent_size = 1024 # should I increase this?

#optimizer and cost function
my_gan_control_LR_generator = 1e-4 #not used for RMSProp   should I reduce these LRs?
my_gan_control_LR_discriminator = 1e-3 #not used for RMSProp
my_gan_control_param_clamp = None #None, 1.0
my_gan_control_relative_vs_meansquared = "MeanSquared" #MeanSquared vs RelativeCost
my_gan_control_generator_optimizer = "Adam" # Adam; RMSProp; anything else it will set to GradientDescent
my_gan_control_discriminator_optimizer = "SGD" # Adam; RMSProp; anything else it will set to GradientDescent
my_control_cost_function = "Modified" #  Wasserstein, Modified, Original
# with Wasserstein on cost displayed is weird and bouncing from negative to positive; review gradient clipping;
# check why it is so small; learning happen however.
# TODO indeed: also wgan_param_clamp must be enabled by this set to Wasserstein

#model configuration
my_three_lines = True
my_alpha = (1, 0.1, 0.05) # : R/F, SUMEcal, Ep
my_alpha_balanced = (1, 1, 1) # 0 multiplier in my_gan_model will apply in this case (my_three_lines = True) on lines other than real/fake
my_gan_lshape = (1, 25, 25, 25)
discriminator_option = 1 # 1 original ,2 = Sofia's 2nd version ,3 = all convolution
generator_option = 2 # 1 original ,2 = Sofia's 2nd version ,3 = all deconvolution
data_normalization = "for_logistic_output"# "for_tanh_output" "for_logistic_output"; else or nothing for relu (as output layer of generator)
my_gan_control_gen_top = "logistic" #"tanh", "logistic", anything else will be lrelu defined in the generator definition todo: in energy_dataset review the mean computation!
my_gan_k = 2 #>0 should I increase this?
my_gan_control_gen_times = 1 #2 it was as in Keras
my_gan_contol_train_gen = True
my_gan_control_trick = True #enabling backprop on generator making discriminator think that is data: trick as in Keras implementation??
inference_only = False #CHANGE IT accordingly!

# settings for inference only
if inference_only:
    my_inference_dir = "inference_results/"
else:
    # settings for training
    my_run_random_prefix = str(int(np.random.randint(1,10000000, 1))) + "_"
    print("################## SIMULATION PREFIX FOR OUTPUT IDENTIFICATION IS: {} ####################".format(my_run_random_prefix))
    my_run_timestamp = time.strftime("%d-%m-%Y-%H-%M")
    res_dir = "results_{}_{}/".format(my_run_timestamp, my_run_random_prefix)
    try:
        os.makedirs(res_dir)
        print("Created directory {}".format(res_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #copy this file into results dir as to track how they were obtained
    this_file_name = "my_gan_control.py" #os.path.basename(__file__)
    destination_file_name = res_dir + my_run_random_prefix + this_file_name
    copyfile(this_file_name, destination_file_name )


# TODO:
# try with uniform noise and check the behaviour
# limiting gradients may help?
# would like to plot discriminator and generator cost function evolution during training check existing call backs, and verify buffers...
# check the backprop in the tree. Apparently we are learning well, but check for inconsistencies
# check the behaviour of the disc cost and the gen cost
# verify that different optimizers are used, eliminatig constant in branch2 def and see that remains correct
# Passare molti dei plot in callback (ora sono nel codice del modello)
# E prima o poi armonizzare i nomi delle variabili di controllo

# # Check the reason for the followking:
# END OF DISCRMINATOR TRAINING: COSTS: Real/Fake cost: 1.39133143425    Ep cost: 0.0122767258435     SUMEcal cost: 1.00333465494e-11
# Epoch 29  [Train  1562/1563 batches, 1.008951902390 cost, 1688.29s]
# INFO:neon.callbacks.callbacks:Epoch 29 complete.  Train Cost 1.392721295357.
# DEBUG:neon.util.persist:serializing object to: 7273710_energy_gan-generator-22-11-2017-16-52].prm
# DEBUG:neon.util.persist:serializing object to: 7273710_energy_gan-discriminator-22-11-2017-16-52].prm
# Exception TypeError: 'super(type, obj): obj must be an instance or subtype of type' in <bound method my_gan_HDF5Iterator.__del__ of <energy_dataset.my_gan_HDF5Iterator object at 0x56805d0>> ignored
# Exception TypeError: 'super(type, obj): obj must be an instance or subtype of type' in <bound method my_gan_HDF5Iterator.__del__ of <energy_dataset.my_gan_HDF5Iterator object at 0x5680650>> ignored
# why energies around 1 are treated worse??? results whenver Er (generated imaged with Epconditiionin is around 1 are worse)

# e perche' passando da tanh a logistic il costo di partenza di Ep and SUMEcal si alza di molto? poi scende ma perche' peggiora??
# in particolare Ep non varia molto ma SUMEcal diventa mostruoso... Che dipenda dalla inizializzazione del Gen? quando cambio top activation questo conta...
# try now also with Relative Cost
# posso provare a :
# normalize to 0-1, logistic out for generator, relative cost, and try with different learning rates

# and after that check the results with data normalizaiotn -1_1 and tanh output
# initialization is important! especially for tha last layers and when I change between activations: around 0 tanh is 0, Logistic is 1/2!!!
# DO NOT forget to review how the alpha are applied: double check that they are multiplied once and only once at the beginning of the backprop so to
# calculate correctly the deltas that will be used then by the optimizer.... or Alphas are set but then used by the optimizer???
# why when energy_p is below one the predictions of SUMEcal are much worse?

#which cost is displayed here: it starts with generator cost and then accunlates and does average, but it mixes disc cost... check!!
# Epoch 0   [Train ...    1/1563 batches, 0.708468735218 cost, 3.38s]
# try with WGAN now
# my_three_lines not needed as alpha = 0 works (but watch out as in noise generation is still used)
# try with uniform generation for generator feeding
# and chech the result increasiong the dimenstionality of the noise
# and uniform with dataset max to 0-1 and sigmoid

# try to decrease the learning rate!! to increase precision in the reconstruction
# and relative cost has any effect now?

# and try with all DCGAN, mabye removin the additional dense layer