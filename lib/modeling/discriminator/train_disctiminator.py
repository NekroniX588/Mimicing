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

class Discriminator():
    def __init__(self, output_dir, name_student, name_teacher, cfg, sizes, boxes):
        self.dis_epoch = 0
        self.dis_test = 0
        self.cfg = cfg
        self.name_student = name_student
        self.name_teacher = name_teacher
        self.output_dir = output_dir
        self.sizes = sizes
        self.boxes = boxes

    def create_discriminator(self):
        if self.cfg.TYPE == 'Medium' and self.sizes[0] == 19*19:
            from lib.modeling.discriminator.MAP.DNet_19 import D_Net
        elif self.cfg.TYPE == 'Medium' and self.sizes[0] == 38*38:
            from lib.modeling.discriminator.MAP.DNet_38 import D_Net
        elif self.cfg.TYPE == 'Residual' and self.sizes[0] == 19*19:
            from lib.modeling.discriminator.MAP.DNet_Residual_Blocks_19 import D_Net
        elif self.cfg.TYPE == 'Invertal' and self.sizes[0] == 19*19:
            from lib.modeling.discriminator.MAP.DNet_Invertal_Blocks_19 import D_Net
        elif self.cfg.TYPE == 'Attention' and self.sizes[0] == 19*19:
            from lib.modeling.discriminator.MAP.DNet_Attention_Blocks_19 import D_Net
        elif self.cfg.TYPE == 'Patch_1' and self.sizes[0] == 19*19:
            from lib.modeling.discriminator.MAP.DNet_Patch_19 import D_Net
        elif self.cfg.TYPE == 'Patch_2' and self.sizes[0] == 19*19:
            from lib.modeling.discriminator.MAP.DNet_Patch2_19 import D_Net

        elif self.cfg.TYPE == 'Vector_Linear':
            from lib.modeling.discriminator.VECTOR.DNet_1x1_Linear import D_Net
        elif self.cfg.TYPE == 'Vector_Conv':
            from lib.modeling.discriminator.VECTOR.DNet_1x1_Conv import D_Net
        elif self.cfg.TYPE == 'Vector_Self':
            from lib.modeling.discriminator.VECTOR.DNet_1x1_self import D_Net
        else:
            print('ERROR. This type of discriminator is not avalible')
            exit()
        DNet = []
        for i,v in enumerate(self.cfg.CHANALS_MAP):
            if self.cfg.TYPE == 'Vector_Self':
                DNet.append(D_Net(v,d_model=v,d_inner=v,n_head=4,d_k=32,d_v=32,dropout=0.2))
            else:
                DNet.append(D_Net(i, v))
        return DNet



    def train_discriminator(self, model, model_mimic, DNet, data_loader_1, data_loader_2, DNet_optim, criterion, pos, use_gpu, logger):
        model.eval()
        model_mimic.eval()
        for param in model.parameters():
            param.requires_grad = False
        for param in model_mimic.parameters():
            param.requires_grad = False
        for D in DNet:
            D.train()

        f = open(logger, 'a')
        print('Weigths of MobileNet and DarkNet froozen')

        if data_loader_2 is None:
            batch_iterator_1 = iter(data_loader_1)
            epoch_size = len(data_loader_1) 
        else:
            batch_iterator_1 = iter(data_loader_1)
            batch_iterator_2 = iter(data_loader_2)
            epoch_size = len(data_loader_1) + len(data_loader_2)

        for iteration in range(self.cfg.NUM_ITERATION):
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
            # forward
            out, FM = model(images, phase='train_mimic')
            All_Loss = 0
            All_Loss_ = []
            # backprop
            if 'Vector' not in self.cfg.TYPE:
                for i, v in enumerate(FM):

                    DNet_optim[i].zero_grad()
                    out = DNet[i](v)
                    label = torch.zeros(images.size(0))
                    Loss = criterion(out, label)
                    Loss.backward()

                    All_Loss_.append(Loss)
                    All_Loss += Loss

                    DNet_optim[i].step()
            else:
                position = pos(out, targets)
                # backprop
                start = 0
                end = 0
                for i in range(len(FM)):
                    
                    DNet_optim[i].zero_grad()

                    inputs = FM[i]
                    start = start + self.sizes[i]*self.boxes[i]
                    # print(start)
                    maper = position[:,end:start].float()
                    answer = maper.view(inputs.size(0),self.sizes[i],-1).sum(dim=-1)>0
                    inputs = inputs.view(inputs.size(0),-1,inputs.size(1))
                    inputs = inputs[answer]
                    # print(inputs.size(0))
                    if inputs.size(0) > 0:
                        if inputs.size(0) > 32:
                            inputs = inputs[:32]
                        out = DNet[i](inputs)
                        label = torch.zeros(inputs.size(0)).cuda()
                        Loss = criterion(out, label)

                        Loss.backward()
                        DNet_optim[i].step()

                        All_Loss_.append(Loss)
                        All_Loss += Loss
                    else:
                        All_Loss_.append(0)
                        All_Loss += 0

                    end = start 

            print('iter ' + repr(iteration) + ' Loss: %.4f ||' % All_Loss + ' Loss0: %.4f||' % All_Loss_[0] + ' Loss1: %.4f||' % All_Loss_[1] + 
                ' Loss2: %.4f||' % All_Loss_[2] + ' Loss3: %.4f||' % All_Loss_[3] + ' Loss4: %.4f||' % All_Loss_[4] + ' Loss5: %.4f||' % All_Loss_[5] + self.name_student)
            f.write('iter ' + repr(iteration) + ' Loss: %.4f ||' % All_Loss + ' Loss0: %.4f||' % All_Loss_[0] + ' Loss1: %.4f||' % All_Loss_[1] + 
                ' Loss2: %.4f||' % All_Loss_[2] + ' Loss3: %.4f||' % All_Loss_[3] + ' Loss4: %.4f||' % All_Loss_[4] + ' Loss5: %.4f||' % All_Loss_[5] + self.name_student + '\n')

            FM = model_mimic(images, phase='feature')
            
            All_Loss = 0
            All_Loss_ = []

            if 'Vector' not in self.cfg.TYPE:
                for i, v in enumerate(FM):

                    DNet_optim[i].zero_grad()
                    out = DNet[i](v)
                    label = torch.ones(images.size(0))
                    Loss = criterion(out, label)
                    Loss.backward()

                    All_Loss_.append(Loss)
                    All_Loss += Loss

                    DNet_optim[i].step()
            else:
                # backprop
                start = 0
                end = 0
                for i in range(len(FM)):
                    
                    DNet_optim[i].zero_grad()

                    inputs = FM[i]
                    start = start + self.sizes[i]*self.boxes[i]
                    # print(start)
                    maper = position[:,end:start].float()
                    answer = maper.view(inputs.size(0),self.sizes[i],-1).sum(dim=-1)>0
                    inputs = inputs.view(inputs.size(0),-1,inputs.size(1))
                    inputs = inputs[answer]
                    # print(inputs.size(0))
                    if inputs.size(0) > 0:
                        if inputs.size(0) > 32:
                            inputs = inputs[:32]
                        out = DNet[i](inputs)
                        label = torch.ones(inputs.size(0)).cuda()
                        Loss = criterion(out, label)

                        Loss.backward()
                        DNet_optim[i].step()

                        All_Loss_.append(Loss)
                        All_Loss += Loss
                    else:
                        All_Loss_.append(0)
                        All_Loss += 0

                    end = start 

            print('iter ' + repr(iteration) + ' Loss: %.4f ||' % All_Loss + ' Loss0: %.4f||' % All_Loss_[0] + ' Loss1: %.4f||' % All_Loss_[1] + 
                ' Loss2: %.4f||' % All_Loss_[2] + ' Loss3: %.4f||' % All_Loss_[3] + ' Loss4: %.4f||' % All_Loss_[4] + ' Loss5: %.4f||' % All_Loss_[5] + self.name_teacher)
            f.write('iter ' + repr(iteration) + ' Loss: %.4f ||' % All_Loss + ' Loss0: %.4f||' % All_Loss_[0] + ' Loss1: %.4f||' % All_Loss_[1] + 
                ' Loss2: %.4f||' % All_Loss_[2] + ' Loss3: %.4f||' % All_Loss_[3] + ' Loss4: %.4f||' % All_Loss_[4] + ' Loss5: %.4f||' % All_Loss_[5] + self.name_teacher + '\n')

        f.close()
        print('discriminator pretrained')
        for k, D in enumerate(DNet):
            torch.save(D.state_dict(), self.output_dir + str(self.dis_epoch) + '_' + str(k) + '.pth')
        self.dis_epoch += 1
        for D in DNet:
            D.eval()

        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def test_discriminator(self, model, model_mimic, DNet, data_loader_1, data_loader_2, pos, use_gpu, logger):
        model.eval()
        model_mimic.eval()

        for D in DNet:
            D.eval()

        f = open(logger, 'a')
        print('Weigths of MobileNet and DarkNet froozen')

        if data_loader_2 is None:
            batch_iterator_1 = iter(data_loader_1)
            epoch_size = len(data_loader_1) 
        else:
            batch_iterator_1 = iter(data_loader_1)
            batch_iterator_2 = iter(data_loader_2)
            epoch_size = len(data_loader_1) + len(data_loader_2)

        acc_s = 0
        all_s = 0
        acc_t = 0
        all_t = 0
        for iteration in iter(range(self.cfg.NUM_ITERATION)):
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

            out, FM = model(images, phase='train_mimic')

            if 'Vector' not in self.cfg.TYPE:
                for i, v in enumerate(FM):

                    out = DNet[i](v)
                    acc_s += sum(out.le(0.5).long())
                    all_s += out.size(0)
            else:
                position = pos(out, targets)
                start = 0
                end = 0
                # backprop
                for i in range(len(FM)):

                    inputs = FM[i]
                    start = start + self.sizes[i]*self.boxes[i]
                    maper = position[:,end:start].float()
                    answer = maper.view(inputs.size(0),self.sizes[i],-1).sum(dim=-1)>0
                    inputs = inputs.view(inputs.size(0),-1,inputs.size(1))
                    inputs = inputs[answer]
                    
                    if inputs.size(0)>0:
                        if inputs.size(0)>32:
                            inputs = inputs[:32]
                        out = DNet[i](inputs)
                        acc_s += sum(out.le(0.5).long())
                        all_s += out.size(0)
                    end = start
                
            FM = model_mimic(images, phase='feature')

            if 'Vector' not in self.cfg.TYPE:
                for i, v in enumerate(FM):

                    out = DNet[i](v)
                    acc_t += sum(out.ge(0.5).long())
                    all_t += out.size(0)
            else:
                start = 0
                end = 0
                # backprop
                for i in range(len(FM)):

                    inputs = FM[i]
                    start = start + self.sizes[i]*self.boxes[i]
                    maper = position[:,end:start].float()
                    answer = maper.view(inputs.size(0),self.sizes[i],-1).sum(dim=-1)>0
                    inputs = inputs.view(inputs.size(0),-1,inputs.size(1))
                    inputs = inputs[answer]
                    
                    if inputs.size(0)>0:
                        if inputs.size(0)>32:
                            inputs = inputs[:32]
                        out = DNet[i](inputs)
                        acc_t += sum(out.ge(0.5).long())
                        all_t += out.size(0)
                    end = start

            log = '\r==>Test: {iter:d}/{size:d}||  Acc_s: {Acc_s:.4f} ||  Acc_t: {Acc_t:.4f}\r'.format(
            Acc_s=acc_s.item()/all_s, Acc_t=acc_t.item()/all_t, iter=iteration+1, size=self.cfg.NUM_ITERATION)
            sys.stdout.write(log)
            sys.stdout.flush()


        print('Discriminator_Acc=',(acc_s.item()+acc_t.item())/(all_s+all_t))
        f.write('Discriminator_Acc='+str((acc_s.item()+acc_t.item())/(all_s+all_t))+'\n')
        model.train()
        for D in DNet:
            D.eval()

   