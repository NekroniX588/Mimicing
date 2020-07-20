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

class Trainer():
    def __init__(self, ):
        pass

    def train_epoch(self, model, data_loader_1, data_loader_2, optimizer, criterion, writer, epoch, use_gpu, logger, loglosses):
        model.train()
        
        if data_loader_2 is None:
            batch_iterator_1 = iter(data_loader_1)
            epoch_size = len(data_loader_1) 
        else:
            batch_iterator_1 = iter(data_loader_1)
            batch_iterator_2 = iter(data_loader_2)
            epoch_size = len(data_loader_1) + len(data_loader_2)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        f = open(logger, 'a')
        g = open(loglosses, 'a')

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
            # forward
            out = model(images, phase='train')

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)

            # some bugs in coco train2017. maybe the annonation bug.
            if loss_l.item() == float("Inf"):
                continue

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            # log per iter
            log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
            prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))),\
            iters=iteration, epoch_size=epoch_size, time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())
            full_log = '|| loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\n'.format(
            loc_loss=loss_l.item(), cls_loss=loss_c.item())

            g.write(full_log)
            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} ||\
        lr: {lr:.6f}\n'.format(lr=lr,time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        f.write(log)
        sys.stdout.write(log)
        sys.stdout.flush()
        f.close()
        g.close()
        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)