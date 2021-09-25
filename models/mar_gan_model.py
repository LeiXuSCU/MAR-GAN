import ntpath
import os
import re
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from models import networks
from utils import gpu_util, data_util


class MarGanModel:
    def __init__(self, opts):
        self.opts = opts

        if self.opts.cuda.enable:
            gpu_count = gpu_util.set_gpu(self.opts.cuda.gpus)
            self.device = torch.device('cuda')
            if self.opts.cuda.cudnn:
                torch.backends.cudnn.benchmark = True
            if gpu_count > 0:
                self.gpu_parallel = True
        else:
            self.device = torch.device('cpu')

        self.net_g = networks.BackboneResnet(self.opts.in_channels, self.opts.model_type)
        if self.gpu_parallel:
            self.net_g = nn.DataParallel(self.net_g)
        self.net_g = self.net_g.to(self.device)

        if self.opts.is_train:
            self.net_d = networks.Discriminator()
            if self.gpu_parallel:
                self.net_d = nn.DataParallel(self.net_d)
            self.net_d = self.net_d.to(self.device)

        self.schedulers = []
        self.optimizers = []
        self.losses = OrderedDict()
        if self.opts.is_train:
            self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=self.opts.learn_rate)
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=self.opts.learn_rate)
            self.optimizers.append(self.optimizer_g)
            self.optimizers.append(self.optimizer_d)
            self.schedulers.append(self.get_scheduler(self.optimizer_g))
            self.schedulers.append(self.get_scheduler(self.optimizer_d))
            if self.opts.pretrained.enable:
                self.load_pretrained_model()
        else:
            self.load_test_model()

    def train(self, data):
        self.net_g.train()
        self.net_d.train()
        # setup data
        from_image, to_image, image_path = data
        real_a = from_image.to(self.device)
        real_b = to_image.to(self.device)

        # update D
        # compute fake images: G(A)
        fake_b = self.net_g(real_a)
        # enable backpropagation for D
        self.set_requires_grad(True)
        # set D's gradients to zero
        self.optimizer_d.zero_grad()
        # calculate gradients for D
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = self.net_d(fake_ab.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = self.net_d(real_ab)
        loss_d_real = self.criterionGAN(pred_real, True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        # update D's weights
        self.optimizer_d.step()

        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(False)
        # set G's gradients to zero
        self.optimizer_g.zero_grad()
        # calculate gradients for G
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = self.net_d(fake_ab)
        loss_g_gan = self.criterionGAN(pred_fake, True)
        loss_g_l2 = self.criterionL2(fake_b, real_b) * 10
        loss_g = loss_g_gan + loss_g_l2
        loss_g.backward()
        # update G's weights
        self.optimizer_g.step()
        self.losses['G_GAN'] = float(loss_g_gan)
        self.losses['G_L2'] = float(loss_g_l2)
        self.losses['D_REAL'] = float(loss_d_real)
        self.losses['D_FAKE'] = float(loss_d_fake)

    def test(self, data):
        self.net_g.eval()
        with torch.no_grad():
            from_image, to_image, image_path = data
            real_a = from_image.to(self.device)
            fake_b = self.net_g(real_a)
            self.test_save(image_path, fake_b)

    def set_requires_grad(self, requires_grad=False):
        if self.net_d is not None:
            for param in self.net_d.parameters():
                param.requires_grad = requires_grad

    def load_pretrained_model(self):
        pretrain_dict = torch.load(self.opts.pretrained.path)
        model_dict = self.basic_save(self.net_g)
        pattern = re.compile(r'(?!fc|conv1)')
        for key in list(pretrain_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = "backbone." + key
                if new_key in model_dict.keys():
                    print("Loading parameter {}".format(new_key))
                    model_dict[new_key] = pretrain_dict[key]
        self.basic_load(self.net_g, model_dict)

    def basic_load(self, net, state_dict):
        if self.gpu_parallel:
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)

    def basic_save(self, net):
        if self.gpu_parallel:
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        return state_dict

    def train_save(self, epoch):
        state_dict = self.basic_save(self.net_g)
        torch.save(state_dict, os.path.join(self.opts.project_root, self.opts.checkpoint_path, '{}.pth'.format(epoch)))

    def test_save(self, image_path, image_data):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        image = data_util.tensor2im(image_data)
        image_name = '%s.png' % name
        save_path = os.path.join(self.opts.project_root, self.opts.result_path, image_name)
        data_util.save_image(image, save_path, aspect_ratio=self.opts.aspect_ratio)

    def get_scheduler(self, optimizer):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + self.opts.epoch_count - self.opts.niter) / float(self.opts.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def print_networks(self):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.net_g.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % ('G', num_params / 1e6))
        if self.opts.is_train:
            num_params = 0
            for param in self.net_d.parameters():
                num_params += param.numel()
            print('[Network %s] Total number of parameters : %.3f M' % ('D', num_params / 1e6))
        print('-----------------------------------------------')

    def get_current_losses(self):
        return self.losses

    def load_test_model(self):
        test_path = os.path.join(self.opts.project_root, self.opts.checkpoint_path,
                                 '{}.pth'.format(self.opts.latest_epoch))
        test_dict = torch.load(test_path)
        model_dict = self.basic_save(self.net_g)
        model_dict.update(test_dict)
        self.basic_load(self.net_g, model_dict)
