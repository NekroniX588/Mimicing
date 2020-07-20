# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + ?Lloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by ? which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

    def forward(self, predictions, targets, mode = None):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            # print(type(targets[idx]))
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0
        # num_pos = pos.sum()
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        
        loss_c = loss_c.view(pos.size(0), pos.size(1))
        # Hard Negative Mining
        loss_c[pos] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True) #new sum needs to keep the same dim
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # print(pos_idx)
        # print('pos_idx',pos_idx.size())
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # print(neg_idx)
        # print('neg_idx',neg_idx.size())
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + ?Lloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        if mode == 'train_mimic':
            return loss_l,loss_c,pos
        else:
            return loss_l,loss_c

class POSdata(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + ?Lloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by ? which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(POSdata, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

    def forward(self, predictions, targets, mode='pos'):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            # print(type(targets[idx]))
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0
        if mode=='pos':
            return pos
        # num_pos = pos.sum()
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        
        loss_c = loss_c.view(pos.size(0), pos.size(1))

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        sizes = [361, 100, 25, 9, 4, 1]
        boxes = [6, 6, 6, 6, 4, 4]

        activation = torch.zeros(6,4)
        start = 0
        end = 0
        for i in range(6):
            start += sizes[i]*boxes[i]
            # print('start',start)
            # print('end',end)
            pos_idxi = pos_idx[0,end:start]
            posi = pos[0,end:start]
            conf_datai = conf_data[0,end:start]
            conf_ti = conf_t[0,end:start]
            # print(conf_ti.size())
            end = start
            # print('end--',end)
            conf_p = conf_datai.view(-1,self.num_classes)
            conf_p = F.softmax(conf_p, dim=-1)
            # print('conf_p',conf_p.size())
            
            targets_weighted = conf_ti
            all_pos = torch.sum(targets_weighted.eq(torch.argmax(conf_p, dim=1)).float())
            all_neg = torch.sum(targets_weighted.ne(torch.argmax(conf_p, dim=1)).float())
            activation[i][0] = all_pos
            activation[i][1] = all_neg
            # print(all_pos)
            # print(all_neg)  


            conf_p = conf_datai[pos_idxi].view(-1,self.num_classes)
            conf_p = F.softmax(conf_p, dim=-1)
            num_pos, num_neg = 0, 0
            if conf_p.size(0) != 0:
                # print('conf_p',conf_p.size())
                # print('conf_p',conf_p)
                # print(torch.argmax(conf_p, dim=1))

                targets_weighted = conf_ti[posi]
                num_pos = torch.sum(targets_weighted.eq(torch.argmax(conf_p, dim=1)).float())
                num_neg = torch.sum(targets_weighted.ne(torch.argmax(conf_p, dim=1)).float())
                # print(num_pos)
                # print(num_neg)
                # print('targets_weighted',targets_weighted) 
            activation[i][2] = num_pos
            activation[i][3] = num_neg
            
        return activation

        # print('pos_idx',pos_idx.size())
        # print('pos_idx',pos_idx[0,:2166].size())
        # print('conf_data',conf_data[0,:2166].size())

        # conf_p = conf_data[pos_idx].view(-1,self.num_classes)
        
        # conf_p = F.softmax(conf_p, dim=-1)
        # print('conf_p',conf_p.size())
        # print('conf_p',conf_p)
        # print(torch.argmax(conf_p, dim=1))

        # targets_weighted = conf_t[(pos).gt(0)]
        # print('targets_weighted',targets_weighted.size())
        # print('targets_weighted',targets_weighted)
        