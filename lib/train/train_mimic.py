import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init
import random
import sys

from lib.utils.timer import Timer

class Traier_mimic():
    def __init__(self, cfg, sizes, boxes, TYPE):
        self.cfg = cfg
        self.sizes = sizes
        self.boxes = boxes
        self.TYPE = TYPE
        if cfg.L_LOSS is not None:  
            if 'L1' in cfg.L_LOSS:
                self.criterion_L = nn.L1Loss()
            elif 'L2' in cfg.L_LOSS:
                self.criterion_L = nn.MSELoss()
            else:
                self.criterion_L = None 
        else:
            self.criterion_L = None 

    def train_mimic_epoch(self, model, model_mimic, DNet, data_loader_1, data_loader_2, optimizer_GENERATOR,\
    DNet_optim, criterion, criterion_GaN, writer, epoch, use_gpu, logger, loglosses):

        f = open(logger, 'a')
        g = open(loglosses, 'a')

        if data_loader_2 is None:
            batch_iterator_1 = iter(data_loader_1)
            epoch_size = len(data_loader_1) 
        else:
            batch_iterator_1 = iter(data_loader_1)
            batch_iterator_2 = iter(data_loader_2)
            epoch_size = len(data_loader_1) + len(data_loader_2)

        loc_loss = 0
        conf_loss = 0
        gan_loss = 0
        if self.criterion_L is not None:
            l_loss = 0
        _t = Timer()

        for iteration in iter(range((epoch_size))):
            if data_loader_2 is not None:
                if random.random() < len(data_loader_1)/epoch_size:
                    try:
                        images, targets = next(batch_iterator_1)
                    except:
                        batch_iterator_1 = iter(data_loader_1)
                        images, targets = next(batch_iterator_1)
                else:
                    try:
                        images, targets = next(batch_iterator_2)
                    except:
                        batch_iterator_2 = iter(data_loader_2)
                        images, targets = next(batch_iterator_2)
            else:
                try:
                    images, targets = next(batch_iterator_1)
                except:
                    batch_iterator_1 = iter(data_loader_1)
                    images, targets = next(batch_iterator_1)
                
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            _t.tic()

            if self.criterion_L is not None:
                FM_real = model_mimic(images, phase='feature')
                loss_L  = 0 

            out, FM_fake = model(images, phase='train_mimic')

            loss_D_s = 0
                       
            optimizer_GENERATOR.zero_grad()

            if 'Vector' not in self.TYPE:
                loss_l, loss_c = criterion(out, targets)

                for k, v in enumerate(FM_fake):
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    label = torch.ones(images.size(0))
                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    result = DNet[k](v)
                    loss_D_s += criterion_GaN(result,label)

                    if self.criterion_L is not None:
                        loss_L += self.criterion_L(v,FM_real[k])
            else:
                loss_l, loss_c, pos = criterion(out, targets, mode = 'train_mimic')
                start = 0
                end = 0
                for i in range(len(FM_fake)):

                    inputs = FM_fake[i]
                    start = start + self.sizes[i]*self.boxes[i]
                    # print(start)
                    maper = pos[:,end:start].float()
                    answer = maper.view(inputs.size(0),self.sizes[i],-1).sum(dim=-1)>0
                    inputs = inputs.view(inputs.size(0),-1,inputs.size(1))
                    inputs = inputs[answer]
                    # print(inputs.size(0))
                    if inputs.size(0) > 0:
                        out = DNet[i](inputs)
                        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        label = torch.ones(inputs.size(0)).cuda()
                        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        loss_D_s += criterion_GaN(out, label)
                    end = start
                    if self.criterion_L is not None:
                        loss_L += self.criterion_L(FM_fake[i],FM_real[i])
                    
            
            # some bugs in coco train2017. maybe the annonation bug.
            if loss_l.item() == float("Inf"):
                continue
            if self.criterion_L is not None:
                loss = loss_l + loss_c + self.cfg.G_ALFA*loss_D_s + self.cfg.L_ALFA*loss_L
            else:
                loss = loss_l + loss_c + self.cfg.G_ALFA*loss_D_s
            loss.backward()

            optimizer_GENERATOR.step()

            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            gan_loss += loss_D_s.item()
            if self.criterion_L is not None:
                l_loss += loss_L.item()
            # log per iter
            if self.criterion_L is not None:
                log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || GAN_loss: {loss_D_s:.4f} L_loss: {L_loss:.4f} || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f} || LOSS: {LOSS:.4f}\r'.format(
                prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))),\
                iters=iteration, epoch_size=epoch_size,time=time, loss_D_s=self.cfg.G_ALFA*loss_D_s.item(),\
                L_loss=self.cfg.L_ALFA*loss_L.item(), loc_loss=loss_l.item(), cls_loss=loss_c.item(), LOSS=loss.item())

                full_log = '|| GAN_loss: {loss_D_s:.4f} L_loss: {L_loss:.4f} || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f} || LOSS: {LOSS:.4f}\n'.format(
                loss_D_s=self.cfg.G_ALFA*loss_D_s.item(), L_loss=self.cfg.L_ALFA*loss_L.item(), loc_loss=loss_l.item(),\
                cls_loss=loss_c.item(), LOSS=loss.item())
            else:
                log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || GAN_loss: {loss_D_s:.4f} || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f} || LOSS: {LOSS:.4f}\r'.format(
                prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))),\
                iters=iteration, epoch_size=epoch_size,time=time, loss_D_s=self.cfg.G_ALFA*loss_D_s.item(),
                loc_loss=loss_l.item(), cls_loss=loss_c.item(), LOSS=loss.item())

                full_log = '|| GAN_loss: {loss_D_s:.4f} || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f} || LOSS: {LOSS:.4f}\n'.format(
                loss_D_s=self.cfg.G_ALFA*loss_D_s.item(), loc_loss=loss_l.item(), cls_loss=loss_c.item(),\
                LOSS=loss.item())

            g.write(full_log)
            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer_GENERATOR.param_groups[0]['lr']
        if self.criterion_L is not None:
            log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} gan_loss: {gan_loss:.4f} l_loss: {l_loss:.4f} || lr: {lr:.6f}\n'.format(
            lr=lr, time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size, gan_loss=gan_loss/epoch_size, l_loss=l_loss/epoch_size)
        else:
            log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} gan_loss: {gan_loss:.4f} || lr: {lr:.6f}\n'.format(
            lr=lr, time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size, gan_loss=gan_loss/epoch_size)
        f.write(log)

        sys.stdout.write(log)
        sys.stdout.flush()
        f.close()
        g.close()
        # log for tensorboard
        writer.add_scalar('Train/gan_loss', gan_loss/epoch_size, epoch)
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        if self.criterion_L is not None:
            writer.add_scalar('Train/l_loss', l_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)

        return gan_loss/epoch_size


   