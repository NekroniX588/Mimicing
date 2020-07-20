import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

import numpy as np
import pandas as pd
from tqdm import tqdm


class Correlation():
    def __init__(self,cfg, sizes, boxes):
        self.cfg = cfg
        self.sizes = torch.tensor(sizes)
        self.boxes = torch.tensor(boxes)

    def caclulate_correlation(self, model, model_mimic, data_loader, logger, criterion, use_gpu):
        model.eval()
        model_mimic.eval()
        batch_iterator = iter(data_loader)

        steps = [0 for i in range(len(self.sizes))]
        mean_corr_c = [[0,0,0] for i in range(len(self.sizes))]
        mean_corr_p = [[0,0,0,0] for i in range(len(self.sizes))]
        f = open(logger, 'a')

        for i in tqdm(range(self.cfg.LEN)):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            out_student, FM_student = model(images, phase='train_mimic')
            FM_teacher = model_mimic(images, phase='feature')
            pos = criterion(out_student, targets).float()
            # print(pos)
            for j in range(len(self.sizes)):
                needed = torch.zeros(self.sizes[j])
                start = sum(self.sizes[:j]*self.boxes[:j])
                for k in range(self.sizes[j]):
                    if sum(pos[0][start + k*self.boxes[j]:start + k*self.boxes[j]+self.boxes[j]]) > 0:
                        needed[k]=1
                needed = needed.bool()  

                chanal = FM_student[j].size(1)   
                S_p = FM_student[j].permute(0,2,3,1).view(-1,chanal)
                T_p = FM_teacher[j].permute(0,2,3,1).view(-1,chanal)
                # print(S_p.size())
                S_p = S_p[needed]
                # print(S_p.size())  
                # print('====================') 
                T_p = T_p[needed]
                dfs = pd.DataFrame(S_p.transpose(0,1).cpu().detach().numpy())
                dft = pd.DataFrame(T_p.transpose(0,1).cpu().detach().numpy())
                st = dfs.corrwith(dft)
                # print(st.mean())
                if not np.isnan(st.mean()):
                    mean_corr_p[j][3] += st.mean()
                    steps[j] += 1
                # print(st.mean())

                S_p = FM_student[j].permute(0,2,3,1).view(-1,chanal)
                T_p = FM_teacher[j].permute(0,2,3,1).view(-1,chanal)

                dfs = pd.DataFrame(S_p.transpose(0,1).cpu().detach().numpy())
                dft = pd.DataFrame(T_p.transpose(0,1).cpu().detach().numpy())
                st = dfs.corrwith(dft)
                mean_corr_p[j][0] += st.mean()
                mean_corr_p[j][1] += st.max()
                mean_corr_p[j][2] += st.min()

                S_c = FM_student[j].view(FM_student[j].size(1),-1)
                T_c = FM_teacher[j].view(FM_teacher[j].size(1),-1)
                dfs = pd.DataFrame(S_c.transpose(0,1).cpu().detach().numpy())
                dft = pd.DataFrame(T_c.transpose(0,1).cpu().detach().numpy())
                st = dfs.corrwith(dft)
                mean_corr_c[j][0] += st.mean()
                mean_corr_c[j][1] += st.max()
                mean_corr_c[j][2] += st.min()

        f.write('Channel||Pixel\n')
        for i in range(6):
            f.write('{ch1:9.5f} {ch2:9.5f} {ch3:9.5f}|| {px1:8.5f} {px2:8.5f} {px3:8.5f} {px4:8.5f} {st:d}\n'.format(
                ch1 = mean_corr_c[i][0]/self.cfg.LEN, ch2 = mean_corr_c[i][1]/self.cfg.LEN, ch3 = mean_corr_c[i][2]/self.cfg.LEN,
                px1 = mean_corr_p[i][0]/self.cfg.LEN, px2 = mean_corr_p[i][1]/self.cfg.LEN, px3 = mean_corr_p[i][2]/self.cfg.LEN, 
                px4 = mean_corr_p[i][3], st = steps[i]))   
        f.close()
        model.train()



    def detection_true(self, model, model_mimic, data_loader, logger, criterion, use_gpu):
        model.eval().cuda()
        model_mimic.eval().cuda()
        batch_iterator = iter(data_loader)
        sizes = torch.tensor([361, 100, 25, 9, 4, 1])
        boxes = torch.tensor([6, 6, 6, 6, 4, 4])
        activations = torch.zeros(6,4).long()
        f = open(logger, 'a')

        for i in tqdm(range(len(batch_iterator))):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            out_student, FM_student = model(images, phase='train_mimic')
            FM_teacher = model_mimic(images, phase='feature')
            activations += criterion(out_student, targets,'bbox').long()
            # print(pos)
        
        f.write('all pos || all neg || num pos || num neg\n')
        for i in range(6):
            f.write('{px1:d} {px2:d} {px3:d} {px4:d}\n'.format(
                px1 = activations[i][0], px2 = activations[i][1], px3 = activations[i][2], px4 = activations[i][3]))   
        f.close()
        model.train()