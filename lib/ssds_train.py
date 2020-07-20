from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import random
import pickle
import time
import shutil
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

from lib.layers import *
from lib.utils.timer import Timer
from lib.test.test import Tester
from lib.test.correlation import Correlation
from lib.train.train import Trainer
from lib.train.train_mimic import Traier_mimic
from lib.modeling.discriminator.train_disctiminator import Discriminator

from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
from utils.nms_wrapper import nms

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self):
        
        self.cfg = cfg
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # Load data
        print('===> Loading data')
        if 'train_mimic' == cfg.PHASE[0] or 'train' == cfg.PHASE[0]:
            self.train_loader_1 = load_data(cfg.DATASET, 'train')
            print(cfg.DATASET.DATASET,len(self.train_loader_1))
            if cfg.DATASET2.DATASET in cfg.DATASET.DATASETS:
                self.train_loader_2 = load_data(cfg.DATASET2, 'train')
                print(cfg.DATASET2.DATASET,len(self.train_loader_1))
            else:
                self.train_loader_2 = None

        self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE1 else None
        self.corr_loader = load_data(cfg.DATASET, 'correlation') if 'correlation' in cfg.PHASE1 else None


        print('===> Building model')
        self.model, self.priorbox, feature_maps = create_model(cfg.MODEL)
        sizes = []
        boxes = []
        for maps in feature_maps:
            sizes.append(maps[0]*maps[1])
        for box in cfg.MODEL.ASPECT_RATIOS:
            boxes.append(len(box)*2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detector = Detect_fast(cfg.POST_PROCESS, self.priors)

        self.Discriminator=Discriminator(cfg.LOG_DIR, cfg.MODEL.NETS, cfg.MODEL_MIMIC.NETS, cfg.DISCTRIMINATOR, sizes, boxes)
        self.Correlation=Correlation(cfg.CORRELATION, sizes, boxes)
        self.Trainer = Trainer()
        self.Traier_mimic = Traier_mimic(cfg.TRAIN_MIMIC, sizes, boxes, cfg.DISCTRIMINATOR.TYPE)
        self.Tester = Tester(cfg.POST_PROCESS)

        if 'train_mimic' == cfg.PHASE[0] or 'correlation' == cfg.PHASE[0]:
            self.model_mimic = create_model(cfg.MODEL_MIMIC)
            self.model_mimic.load_state_dict(torch.load(cfg.MODEL_MIMIC.WEIGHTS))

            self.DNet = self.Discriminator.create_discriminator()

        

        # Utilize GPUs for computation
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            cudnn.benchmark = True
            # self.model_mimic = torch.nn.DataParallel(self.model_mimic)
            # for i in range(len(self.DNet)):
            #     self.DNet[i] = torch.nn.DataParallel(self.DNet[i])
            self.model.cuda()
            self.priors.cuda()
            if 'train_mimic' == cfg.PHASE[0] or 'correlation' == cfg.PHASE[0]: 
                self.model_mimic.cuda()
                for i in range(len(self.DNet)):
                    self.DNet[i] = self.DNet[i].cuda()

        # if 'train_mimic' == cfg.PHASE[0]: 
            # print('Model mimim architectures:\n{}\n'.format(self.model_mimic))
            # for i in range(len(self.DNet)):
            #   print('Hello')
            #   print(self.DNet[i])
        # print('Parameters and size:')
        # for name, param in self.model.named_parameters():
        #     print('{}: {}'.format(name, list(param.size())))

        # print trainable scope
        # print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        if 'train_mimic' == cfg.PHASE[0] or 'correlation' == cfg.PHASE[0]:
            self.optimizer = self.configure_optimizer(self.model, cfg.TRAIN.OPTIMIZER)
            self.DNet_optim = []
            for i in range(len(self.DNet)):
                self.DNet_optim.append(self.configure_optimizer(self.DNet[i], cfg.DISCTRIMINATOR.OPTIMIZER))
            self.optimizer_GENERATOR = self.configure_optimizer(self.model, cfg.TRAIN_MIMIC.OPTIMIZER)
            self.exp_lr_scheduler_g = self.configure_lr_scheduler(self.optimizer_GENERATOR, cfg.TRAIN.LR_SCHEDULER)
        else:
            self.optimizer = self.configure_optimizer(self.model, cfg.TRAIN.OPTIMIZER)
            
        self.phase = cfg.PHASE
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.criterion = MultiBoxLoss(cfg.MATCHER, self.priors, self.use_gpu)
        self.criterion_GaN = nn.BCELoss()
        self.pos = POSdata(cfg.MATCHER, self.priors, self.use_gpu)

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        if not os.path.exists(cfg.LOG_DIR):
            os.mkdir(cfg.LOG_DIR)
        shutil.copyfile('./lib/utils/config_parse.py', cfg.LOG_DIR+'hiperparameters.py')
        a = os.listdir(cfg.LOG_DIR)
        for i in range(1,100):
            if not 'Correlation_'+str(i)+'.txt' in a:
                self.logger = cfg.LOG_DIR+'Correlation_'+str(i)+'.txt'
                self.loglosses = cfg.LOG_DIR+'Correlation_loss_'+str(i)+'.txt'
                break
        f = open(self.logger, 'w')
        f.close()
        f = open(self.loglosses, 'w')
        f.close()
        self.output_dir = cfg.LOG_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX
        self.model.loc.apply(self.weights_init)
        self.model.conf.apply(self.weights_init)
        self.model.extras.apply(self.weights_init)
        if 'train' == cfg.PHASE[0]:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)

    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        resume_scope = self.cfg.TRAIN.RESUME_SCOPE
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            start_epoch = self.initialize()

        # export graph for the model, onnx always not works
        # self.export_graph()
        if self.phase[0] == 'train':
        # warm_up epoch
            self.model.train()
            warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
            for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
                #learning rate
                sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
                if epoch > warm_up:
                    self.exp_lr_scheduler.step(epoch-warm_up)
                # if epoch==145:
                #     self.optimizer = optim.SGD(self.model.parameters(), lr=0.001,
                #         momentum=0.9, weight_decay=0.0001)
                #     self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
                if 'train' in cfg.PHASE:
                    self.Trainer.train_epoch(self.model,  self.train_loader_1, self.train_loader_2, self.optimizer, self.criterion,\
                    self.writer, epoch, self.use_gpu, self.logger, self.loglosses)
                if 'test' in cfg.PHASE1 and (epoch % cfg.TEST.STEP == 0 or epoch > cfg.TEST.EPOCH):
                    self.Tester.test_fast_nms(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu, self.writer, epoch)

                if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                    self.save_checkpoints(epoch)


        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        if self.phase[0] == 'train_mimic':
            if not previous:
                self.Discriminator.train_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1, self.train_loader_2,\
                self.DNet_optim, self.criterion_GaN, self.pos, self.use_gpu, self.logger)
                self.Discriminator.test_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1, self.train_loader_2,\
                self.pos, self.use_gpu, self.logger)
            else:
                for k, D in enumerate(self.DNet):
                    D.load_state_dict(torch.load(self.output_dir + '0_' + str(k) + '.pth'))
                    print('Discriminator loaded')

            for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
                #learning rate
                sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
                with open(self.logger, 'a') as f:
                    f.write('Epoch ' + str(epoch)+'\n')

                if epoch > warm_up:
                    self.exp_lr_scheduler.step(epoch-warm_up)
                    self.exp_lr_scheduler_g.step(epoch-warm_up)

                if epoch <= cfg.TRAIN_MIMIC.EPOCHS:
                    gan_loss = self.Traier_mimic.train_mimic_epoch(self.model, self.model_mimic, self.DNet, self.train_loader_1,\
                    self.train_loader_2, self.optimizer_GENERATOR, self.DNet_optim, self.criterion, self.criterion_GaN, self.writer,\
                    epoch, self.use_gpu, self.logger, self.loglosses)

                    self.Discriminator.test_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1, self.train_loader_2,\
                    self.pos, self.use_gpu, self.logger)

                    if cfg.TRAIN_MIMIC.TYPE_TRAINING == 'THRESHOLD':
                        if gan_loss < cfg.TRAIN_MIMIC.G_THRESHOLD:
                            self.Discriminator.train_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1,\
                            self.train_loader_2, self.DNet_optim, self.criterion_GaN, self.pos, self.use_gpu, self.logger)

                            self.Discriminator.test_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1,\
                            self.train_loader_2, self.pos, self.use_gpu, self.logger)

                    if cfg.TRAIN_MIMIC.TYPE_TRAINING == 'PERIOD':
                        if epoch % cfg.TRAIN_MIMIC.G_PERIOD == 0:
                            self.Discriminator.train_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1,\
                            self.train_loader_2, self.DNet_optim, self.criterion_GaN, self.pos, self.use_gpu, self.logger)

                            self.Discriminator.test_discriminator(self.model, self.model_mimic, self.DNet, self.train_loader_1,\
                            self.train_loader_2, self.pos, self.use_gpu, self.logger)

                if epoch > cfg.TRAIN_MIMIC.EPOCHS:
                    self.Trainer.train_epoch(self.model,  self.train_loader_1, self.train_loader_2, self.optimizer, self.criterion,\
                    self.writer, epoch, self.use_gpu, self.logger, self.loglosses)

                if (epoch % cfg.TEST.STEP == 0 or epoch > cfg.TEST.EPOCH) and self.test_loader is not None:
                    self.Tester.test_fast_nms(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu, self.writer, epoch)

                if epoch % cfg.CORRELATION.STEP == 0 and self.corr_loader is not None:
                    self.Correlation.caclulate_correlation(self.model, self.model_mimic, self.corr_loader, self.logger, self.pos, self.use_gpu)

                if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                    self.save_checkpoints(epoch)

    def test_model(self):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
                    sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.cfg.TEST.TEST_SCOPE[1]))
                    self.resume_checkpoint(resume_checkpoint)
                    if 'eval' in cfg.PHASE:
                        self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
                    if 'test' in cfg.PHASE:
                        self.test_fast_nms(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
                    if 'visualize' in cfg.PHASE:
                        self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
                    if 'correlation' in cfg.PHASE:
                        self.caclulate_correlation(self.model, self.model_mimic, self.corr_loader, self.logger, self.pos, self.use_gpu)
        else:
            sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            if 'eval' in cfg.PHASE:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, 0, self.use_gpu)
            if 'test' in cfg.PHASE:
                self.test_fast_nms(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, 0,  self.use_gpu)
            if 'correlation' in cfg.PHASE:
                self.caclulate_correlation(self.model, self.model_mimic, self.corr_loader, self.logger, self.pos, self.use_gpu)

    def configure_optimizer(self, model, cfg):
        if cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                        betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer, cfg):
        if cfg.SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'SGDR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler


    def export_graph(self):
        self.model.train(False)
        dummy_input = Variable(torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).cuda()
        # Export the model
        torch_out = torch.onnx._export(self.model,             # model being run
                                       dummy_input,            # model input (or a tuple for multiple inputs)
                                       "graph.onnx",           # where to save the model (can be a file or file-like object)
                                       export_params=True)     # store the trained parameter weights inside the model file
        # if not os.path.exists(cfg.EXP_DIR):
        #     os.makedirs(cfg.EXP_DIR)
        # self.writer.add_graph(self.model, (dummy_input, ))


def train_model():
    s = Solver()
    s.train_model()
    return True

def test_model():
    s = Solver()
    s.test_model()
    return True
