import numpy as np


# load up the data set
X= np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
mean = np.mean(X, axis=0, keepdims=True)
max_elem = np.max(np.abs(X))
print(np.max(np.abs(X)),'max abs element')
print(np.min(X),'min element')

# it seems to be needed as the generator output is tanh! Note that Initial normalizaion not done Energies (labels)
X = (X - mean)/max_elem
print(X.shape, 'X shape')
print(np.max(X),'max element after normalisation')
print(np.min(X),'min element after normalisation')


'''
To check my changes to Neon:

-mobl:neon azanetti$ git diff --name-only 0f158d0cc214bb4a8bd751b43885774f4e7a23e3
examples/CERN_3DGAN/__init__.py
examples/CERN_3DGAN/neon/__init__.py
examples/CERN_3DGAN/neon/energy_dataset.py
examples/CERN_3DGAN/neon/energy_gan.py
examples/CERN_3DGAN/neon/my_gan_control.py
examples/CERN_3DGAN/neon/my_gan_costs.py
examples/CERN_3DGAN/neon/my_gan_hdf5_build.py
examples/CERN_3DGAN/neon/my_gan_inference.py
examples/CERN_3DGAN/neon/my_gan_layers.py
examples/CERN_3DGAN/neon/my_gan_model.py
examples/CERN_3DGAN/neon/my_gan_reading_images_from_dataset.py
examples/CERN_3DGAN/neon/other_files/__init__.py
examples/CERN_3DGAN/neon/other_files/energy_train.py
examples/CERN_3DGAN/neon/other_files/gan_defs.py
examples/CERN_3DGAN/neon/other_files/gan_layers.py
examples/CERN_3DGAN/neon/other_files/generate_images.py
examples/CERN_3DGAN/utils/data_info.py
examples/CERN_3DGAN/utils/gaussian.py
examples/CERN_3DGAN/utils/lcd_utils.py
examples/__init__.py
examples/gan/mnist_dcgan.py
neon/backends/nervanacpu.py
neon/backends/nervanamkl.py
neon/callbacks/callbacks.py
neon/layers/container.py
neon/layers/layer.py
neon/models/model.py
neon/transforms/cost.py
-mobl:neon azanetti$ 

In detail:

git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/backends/nervanacpu.py
git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/backends/nervanamkl.py
git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/callbacks/callbacks.py
git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/layers/container.py
git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/layers/layer.py
git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/models/model.py
git diff 0f158d0cc214bb4a8bd751b43885774f4e7a23e3 neon/transforms/cost.py
'''