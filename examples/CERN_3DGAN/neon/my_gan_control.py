import numpy as np
import time

# control parameters of my_gan
res_dir = "results/"
my_debug = True
my_three_lines = True
my_alpha = (6, 2, 1)
my_alpha_balanced = (1, 1, 1) # 0 multiplier in my_gan_model will apply in this case (my_three_lines = True) on lines other than real/fake
my_gan_lshape = (1, 25, 25, 25)
my_use_hdf5_iterator = True
generator_option_1 = False
discriminator_option_1 = False
save_training_progress = False
plot_matrix = True
my_xavier = False # with True will lead to NANs in discriminator fake/real output. Why?
my_xavier_gen = False
my_gan_control_batch_size = 128
my_gan_control_nb_epochs = 30
my_gan_control_latent_size = 256
my_gan_control_LR = 1e-6
my_compute_all_costs = True

my_control_gan_Wasserstein = False # with this on cost displayed is 0.0000000;
# check why it is so small and learning does not happen.
# Maybe TopLayer is not correct or other tweaks must be enabled by this flag
# TODO indeed: also wgan_param_clamp must be enabled by this

my_run_random_prefix = str(np.random.randint(1,10000000, 1)) + "_"
print("################## SIMULATION PREFIX FOR OUTPUT IDENTIFICATION IS: {} ####################".format(my_run_random_prefix))
timestamp = time.strftime("%d-%m-%Y-%H-%M")

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
generator_option_1 = False
discriminator_option_1 = True 
save_training_progress = False
plot_matrix = True
my_xavier = False # with True will lead to NANs. why?
my_gan_control_batch_size = 128
my_gan_control_nb_epochs = 30
my_gan_control_latent_size = 256
my_gan_control_LR = 5e-4
my_control_gan_Wasserstein = True

'''


# TODO: list below
# Esplicitare tutti I costi (fake/real, gen, aux ep, aux SUMecal) e
# Tracciare andamento durante il training; vorrei plottare il tutto ogni tot iterazioni
# Introdurre uno switch/parametro che permette di rendere il modello WGAN
# Vorrei provare con gen/discr fatti solo di deconv/conv, magari con dropout, e vedere cosa succede.
# Passare molti dei plot in callback (ora sono nel codice del modello)
# E prima o poi armonizzare i nomi delle variabili di controllo
