# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division
import logging
import numpy as np
from PIL import Image
from neon.callbacks.callbacks import Callback
from my_gan_control import *
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.exception("GANPlotCallback requires matplotlib to be installed")
    raise(e)
try:
    from scipy.signal import medfilt
except ImportError as e:
    logger.exception("GANPlotCallback requires scipy to be installed")
    raise(e)


class myGANPlotCallback(Callback):
    """
    Create PNG plots of samples from a GAN model.

    Arguments:
        filename (string): Filename prefix for output PNGs.
        hw (int): Height and width of the images.
        nchan (int): number of channels.
        num_saples (int): how many samples to show from traning and generated data.
        sym_range (bool): pixel value [-1, 1] or [0, 1].
        padding (int): number of pixels to pad in output images.
        plot_width (int): width of output images.
        plot_height (int): height of output images.
        dpi (float): dots per inch.
        font_size (int): font size of labels.
        epoch_freq (int): number of epochs per plotting callback.
    """
    def __init__(self, filename, hw=25, nchan=1, num_samples=64, sym_range=False, padding=2,
                 plot_width=1200, plot_height=600, dpi=60., font_size=10, epoch_freq=1):
        super(myGANPlotCallback, self).__init__(epoch_freq=epoch_freq)
        self.filename = filename
        self.hw = hw
        self.nchan = nchan
        self.num_samples = num_samples
        self.padding = padding
        self.sym_range = sym_range
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.dpi = dpi
        self.font_size = font_size

    def _value_transform(self, batch):
        if self.nchan == 1:
            batch = 1. - batch
        else:
            if self.sym_range:
                batch = (batch + 1.) / 2.
        return batch

    def _shape_transform(self, batch):
        assert self.nchan * self.hw * self.hw  * self.hw == batch.shape[0], "wrong image size specified"
        assert self.num_samples <= batch.shape[1], "number of samples must not exceed batch size"

        nrow = int(np.ceil(np.sqrt(self.num_samples)))
        ncol = int(np.ceil(1. * self.num_samples / nrow))
        width = ncol*(self.hw+self.padding)-self.padding
        height = nrow*(self.hw+self.padding)-self.padding

        batch = batch[:, :self.num_samples]
        batch = batch.reshape(self.nchan, self.hw, self.hw, self.hw, self.num_samples)
        batch = np.swapaxes(np.swapaxes(np.swapaxes(batch, 0, 1), 1, 2), 2, 3)

        canvas = np.ones([height, width, self.nchan])
        for i in range(self.num_samples):
            irow, icol, step = i % nrow, i // nrow, self.hw + self.padding
            canvas[irow*step:irow*step+self.hw, icol*step:icol*step+self.hw, :] = \
                batch[:, :, 12, ::-1, i] / np.amax(batch[:, :, 12, ::-1, i])
        if self.nchan == 1:
            canvas = canvas.reshape(height, width)
        return canvas

    def on_epoch_end(self, callback_data, model, epoch):
        # convert to numpy arrays
        data_batch = model.data_batch.get()
        noise_batch = model.noise_batch.get()
        # value transform
        data_batch = self._value_transform(data_batch)
        noise_batch = self._value_transform(noise_batch)
        # shape transform
        data_canvas = self._shape_transform(data_batch)
        noise_canvas = self._shape_transform(noise_batch)
        # plotting options
        im_args = dict(interpolation="nearest", vmin=0., vmax=1.)
        if self.nchan == 1:
            im_args['cmap'] = plt.get_cmap("gray")
        fname = self.filename + mgc_run_random_prefix + '_data_'+'{:03d}'.format(epoch) + '.png'
        Image.fromarray(np.uint8(data_canvas*255)).convert('RGB').save(fname)
        fname = self.filename + mgc_run_random_prefix + '_noise_'+'{:03d}'.format(epoch) + '.png'
        Image.fromarray(np.uint8(noise_canvas*255)).convert('RGB').save(fname)
        #better: image = (image * 255).round().astype(np.uint8)?

        # plot logged WGAN costs if logged
        if model.cost.costfunc.func == 'wasserstein':
            giter = callback_data['gan/gen_iter'][:]
            nonzeros = np.where(giter)
            giter = giter[nonzeros]
            cost_dis = callback_data['gan/cost_dis'][:][nonzeros]
            w_dist = medfilt(np.array(-cost_dis, dtype='float64'), kernel_size=101)
            plt.figure(figsize=(400/self.dpi, 300/self.dpi), dpi=self.dpi)
            plt.plot(giter, -cost_dis, 'k-', lw=0.25)
            plt.plot(giter, w_dist, 'r-', lw=2.)
            plt.title(self.filename, fontsize=self.font_size)
            plt.xlabel("Generator Iterations", fontsize=self.font_size)
            plt.ylabel("Wasserstein estimate", fontsize=self.font_size)
            plt.margins(0, 0, tight=True)
            plt.savefig(self.filename+'_training.png', bbox_inches='tight')
            plt.close()


class myGANCostCallback(Callback):
    """
    Callback for computing average training cost periodically during training.
    """
    def __init__(self):
        super(myGANCostCallback, self).__init__(epoch_freq=1)

    def on_train_begin(self, callback_data, model, epochs):
        """
        Called when training is about to begin

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
            epochs (int): Total epochs
        """
        # preallocate space for the number of minibatches in the whole run
        points = callback_data['config'].attrs['total_minibatches']
        callback_data.create_dataset("gan/gen_iter", (points,))
        callback_data.create_dataset("gan/cost_dis_Ep", (points,))
        callback_data.create_dataset("gan/cost_dis_SUMEcal", (points,))
        callback_data.create_dataset("gan/cost_gen", (points,))
        callback_data.create_dataset("gan/cost_dis", (points,))

        # clue in the data reader to use the 'minibatch' time_markers
        callback_data['gan/gen_iter'].attrs['time_markers'] = 'minibatch'
        callback_data['gan/cost_dis_Ep'].attrs['time_markers'] = 'minibatch'
        callback_data['gan/cost_dis_SUMEcal'].attrs['time_markers'] = 'minibatch'
        callback_data['gan/cost_gen'].attrs['time_markers'] = 'minibatch'
        callback_data['gan/cost_dis'].attrs['time_markers'] = 'minibatch'

    def on_minibatch_end(self, callback_data, model, epoch, minibatch):
        """
        Log GAN cost data. Called when minibatch is about to end

        Arguments:
            callback_data (HDF5 dataset): shared data between callbacks
            model (Model): model object
            epoch (int): index of current epoch
            minibatch (int): index of minibatch that is ending
        """
        if model.current_batch == model.last_gen_batch and model.last_gen_batch > 0:
            mbstart = callback_data['time_markers/minibatch'][epoch - 1] if epoch > 0 else 0
            callback_data['gan/gen_iter'][mbstart + minibatch] = model.gen_iter
            callback_data['gan/cost_dis'][mbstart + minibatch] = model.cost_dis[0]
            callback_data['gan/cost_dis_Ep'][mbstart + minibatch] = model.cost_dis_Ep
            callback_data['gan/cost_dis_SUMEcal'][mbstart + minibatch] = model.cost_dis_SUMEcal[0]
            callback_data['gan/cost_gen'][mbstart + minibatch] = model.cost_gen[0]

        if (model.current_batch + mgc_data_saving_freq) % mgc_data_saving_freq == 0:
            fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), my_gan_results_dir)
            plfname = mgc_run_random_prefix + os.path.splitext(os.path.basename(__file__))[
                0] + "-" + mgc_timestamp + "-" + "-" + \
                      'Epoch {}'.format(model.epoch_index) + '_[' + 'batch_n_{}'.format(model.current_batch) + ']_GAN_COST_FUNCTIONS'
            plfname_1 = mgc_run_random_prefix + os.path.splitext(os.path.basename(__file__))[
                0] + "-" + mgc_timestamp + "-" + "-" + \
                      'Epoch {}'.format(model.epoch_index) + '_[' + 'batch_n_{}'.format(model.current_batch) + ']_GAN_AUX_FUNCTIONS'
            plt_filename = os.path.join(fdir, plfname)
            plt_filename_1 = os.path.join(fdir, plfname_1)

            t = np.array(range(0, callback_data['config'].attrs['total_minibatches']))
            pdisc = np.array(callback_data['gan/cost_dis'])
            pEp = np.array(callback_data['gan/cost_dis_Ep'])
            pSUMEcal = np.array(callback_data['gan/cost_dis_SUMEcal'])
            pgen = np.array(callback_data['gan/cost_gen'])

            interval_right = 1000 if model.current_batch < 1000 else model.current_batch
            interval_left = 0 #if model.current_batch > 1000 else model.current_batch - 1000

            plt.figure()
            plt.title("Cost functions at iteration n.:{0:d}\n".format(model.current_batch))
            leg_disc = plt.plot(t[interval_left:interval_right], pdisc[interval_left:interval_right], linestyle='None', marker='v', color='r', label=u"Disc")
            leg_gen = plt.plot(t[interval_left:interval_right], pgen[interval_left:interval_right], linestyle='None', marker='<', color='b', label=u"Gen")
            plt.xlabel('minibatch')
            plt.ylabel('GAN costs')
            ylow = np.min(np.minimum(pdisc[interval_left:interval_right], pgen[interval_left:interval_right]))
            yhigh = np.max(np.maximum(pdisc[interval_left:interval_right], pgen[interval_left:interval_right]))
            plt.axis([interval_left, interval_right, ylow, yhigh])
            plt.legend()
            plt.grid(True)
            plt.savefig(plt_filename)
            plt.close()

            plt.figure()
            plt.title("Cost functions at iteration n.:{0:d}\n".format(model.current_batch))
            leg_cost_Ep = plt.plot(t[interval_left:interval_right], pEp[interval_left:interval_right],linestyle='None', marker='v', color='g', label=u"Ep")
            leg_cost_SUMEcal = plt.plot(t[interval_left:interval_right], pSUMEcal[interval_left:interval_right], linestyle='None', marker='<',color='y',
                                        label=u"SUMEcal")
            plt.xlabel('minibatch')
            plt.ylabel('AUX costs')
            ylow = np.min(np.minimum(pEp[interval_left:interval_right], pSUMEcal[interval_left:interval_right]))
            yhigh = np.max(np.maximum(pEp[interval_left:interval_right], pSUMEcal[interval_left:interval_right]))
            plt.axis([interval_left, interval_right, ylow, yhigh])
            plt.legend()
            plt.grid(True)
            plt.savefig(plt_filename_1)
            plt.close()
            print("\nSaved cost functions profile to {}".format(plt_filename))






