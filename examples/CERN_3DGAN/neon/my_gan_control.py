# control parameters of my_gan
res_dir = "results/"
my_debug = True
my_three_lines = False
my_alpha = (6, 2, 1)
my_alpha_balanced = (1, 1, 1) # 0 multiplier in my_gan_model will apply in this case on lines other than real/fake
my_gan_lshape = (1,25,25,25)
my_use_hdf5_iterator = True
generator_option_1 = True
save_training_progress = False
plot_matrix = True
my_xavier = False # with True will lead to NANs. why?
