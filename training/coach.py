"""
Created by Xu Lei on 2021/09/20 21:30.
E-mail address is leixu@stu.scu.edu.cn.
Copyright  2021 Xu Lei. SCU. All Rights Reserved.
"""
import time

from torch.utils.data import DataLoader
from datasets.aligned_dataset import AlignedDataset
from models.mar_gan_model import MarGanModel


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.model = MarGanModel(self.opts)
        self.model.print_networks()

        self.dataset = AlignedDataset(opts)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.opts.batch_size,
                                     shuffle=True,
                                     num_workers=int(self.opts.workers))
        print('Total training images = {}'.format(len(self.dataloader)))

    def train(self):
        total_iteration = 0

        for epoch in range(self.opts.epoch_count,
                           self.opts.niter + self.opts.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iteration = 0

            iter_data_time = time.time()
            for i, data in enumerate(self.dataloader):
                iter_start_time = time.time()
                loss_print_interval = 10
                data_loading_time = 0
                if total_iteration % loss_print_interval == 0:
                    data_loading_time = int(round(iter_start_time * 1000)) - int(round(iter_data_time * 1000))

                total_iteration += self.opts.batch_size
                epoch_iteration += self.opts.batch_size
                self.model.train(data)

                if total_iteration % loss_print_interval == 0:
                    losses = self.model.get_current_losses()
                    computational_time = \
                        (int(round(time.time() * 1000)) - int(round(iter_start_time * 1000))) / self.opts.batch_size
                    print('----------------- LOSS INFO START -----------------')
                    print('epoch: %d, iteration: %d' % (epoch, epoch_iteration))
                    print(
                        'computational_time: %d ms, data_loading_time: %d ms' % (computational_time, data_loading_time))
                    message = ''
                    for k, v in losses.items():
                        message += '%s: %.3f, ' % (k, v)
                    print(message)
                    print('----------------- LOSS INFO END -----------------')

                iter_data_time = time.time();
            if epoch % 5 == 0:
                print('saving the model at the end of epoch %d, total_iterations %d' % (epoch, total_iteration))
                self.model.train_save(self.opts.latest_epoch)
                self.model.train_save(epoch)

            print('End of epoch %d / %d \t Time Taken: %d ms' % (
                epoch, self.opts.niter + self.opts.niter_decay,
                int(round(time.time() * 1000)) - int(round(epoch_start_time * 1000))))
            self.model.update_learning_rate()

    def test(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opts.num_test:
                break
            self.model.test(data)