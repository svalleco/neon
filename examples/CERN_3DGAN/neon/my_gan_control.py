import numpy as np
import time
import os, errno
from shutil import copyfile, copy2

# control parameters of my_gan
#debugging an printing
my_debug = True
plot_matrix = True
my_gan_control_print_tensor_examples = True
my_compute_all_costs = True
save_training_progress = False #hdf files
my_gan_control_save_prm = False #prm files

#data mng
my_use_hdf5_iterator = True
data_saving_freq = 1

#Initializations
my_xavier_discr = False # with True will lead to NANs in discriminator fake/real output. Why?
my_xavier_gen = False
my_gaussian_scale_init_for_generator = 0.001
my_gaussian_scale_init_for_discriminator = 0.01

#duration and batchsize, latent size
my_gan_control_batch_size = 4
my_gan_control_nb_epochs = 30
my_gan_control_latent_size = 1024

#optimizer and cost function
my_gan_control_LR_generator = 1e-4 #not used for RMSProp
my_gan_control_LR_discriminator = 1e-3 #not used for RMSProp
my_gan_control_param_clamp = None
my_gan_control_relative_vs_meansquared = "RelativeCost" #RelativeCost vs MeanSquared
my_gan_control_generator_optimizer = "Adam" # Adam; RMSProp; anything else it will set to GradientDescent
my_gan_control_discriminator_optimizer = "SGD" # Adam; RMSProp; anything else it will set to GradientDescent
my_control_cost_function = "Modified" #  Wasserstein, Modified, Original
# with Wasserstein on cost displayed is weird and bouncing from negative to positive; review gradient clipping;
# check why it is so small and learning does not happen.
# Maybe TopLayer is not correct or other tweaks must be enabled by this flag
# TODO indeed: also wgan_param_clamp must be enabled by this set to Wasserstein

#model configuration
my_three_lines = True
my_alpha = (10, 2, 1)
my_alpha_balanced = (1, 1, 1) # 0 multiplier in my_gan_model will apply in this case (my_three_lines = True) on lines other than real/fake
my_gan_lshape = (1, 25, 25, 25)
generator_option = 2 # 1 original ,2 = Sofia's 2nd version ,3 = all deconvolution
discriminator_option = 1 # 1 original ,2 = Sofia's 2nd version ,3 = all convolution
# NB this below requires manually setting of gen output layer accordingly
data_normalization = "no"# """for_tanh_output" # "for_tanh_output" "for_logistic_output"; else or nothing for relu (as output layer of generator)
my_gan_k = 1 #>0
my_gan_control_gen_times = 1 #2 it was as in Keras
my_gan_contol_train_gen = False
my_gan_control_trick = False #enabling backprop on generator making discriminator think that is data: trick as in Keras implementation??
inference_only = False #CHANGE IT accordingly!

# settings for inference only
if inference_only:
    my_inference_dir = "inference_results/"
else:
    # settings for training
    my_run_random_prefix = str(int(np.random.randint(1,10000000, 1))) + "_"
    print("################## SIMULATION PREFIX FOR OUTPUT IDENTIFICATION IS: {} ####################".format(my_run_random_prefix))
    timestamp = time.strftime("%d-%m-%Y-%H-%M")
    res_dir = "results_{}_{}/".format(timestamp, my_run_random_prefix)
    try:
        os.makedirs(res_dir)
        print("Created directory {}".format(res_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #copy this file into results dir as to track how they were obtained
    this_file_name = "my_gan_control.py" #os.path.basename(__file__)
    destination_file_name = res_dir + this_file_name
    copyfile(this_file_name, destination_file_name )


'''
best parameters so far:

# control parameters of my_gan
res_dir = "results/"
my_debug = True
my_three_lines = True
my_alpha = (6, 2, 1)
my_alpha_balanced = (1, 1, 1) # 0 multiplier in my_gan_model will apply in this case on lines other than real/fake
my_gan_lshape = (1, 25, 25, 25)
my_use_hdf5_iterator = True
generator_option_1 = False (=2)
discriminator_option_1 = True 
save_training_progress = False
plot_matrix = True
my_xavier = False # with True will lead to NANs. why?
my_gan_control_batch_size = 128
my_gan_control_nb_epochs = 30
my_gan_control_latent_size = 256
my_gan_control_LR = 5e-4
my_control_cost_function = False

'''


# TODO:
# Esplicitare tutti I costi (fake/real, gen, aux ep, aux SUMecal) e
# Tracciare andamento durante il training; vorrei plottare il tutto ogni tot iterazioni
# Introdurre uno switch/parametro che permette di rendere il modello WGAN
# Vorrei provare con gen/discr fatti solo di deconv/conv, magari con dropout, e vedere cosa succede.
# Passare molti dei plot in callback (ora sono nel codice del modello)
# E prima o poi armonizzare i nomi delle variabili di controllo
