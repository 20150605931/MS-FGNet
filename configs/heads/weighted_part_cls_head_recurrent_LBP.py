from .cls_head import ClsHead
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..losses import *
from core.evaluations import Accuracy
from .mamba import Mamba, MambaConfig

# # 配置标识和超参数
# USE_MAMBA =1
# DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM =0
# # 设定所用设备
# Device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

def pairwise_ranking_loss(preds, size_average = True):
    """
        preds:
            list of scalar Tensor.
            Each value represent the probablity of each class
                e.g) class = 3
                    preds = [logits1[class], logits2[class]]
    """
    if len(preds) <= 1:
        return torch.zeros(1).cuda()
    else:
        losses = []
        for pred in preds:
            loss = []
            for i in range(len(pred)-1):
                rank_loss = (pred[i]-pred[i+1] + 0.25).clamp(min = 0)
                # 0.05 margin
                loss.append(rank_loss)
            loss = torch.sum(torch.stack(loss))
            losses.append(loss)
        losses = torch.stack(losses)
        if size_average:
            losses = torch.mean(losses)
        else:
            losses = torch.sum(losses)
        return losses


class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x

class WeightedPartClsHead_recurrent_LBP(ClsHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 part_num,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(WeightedPartClsHead_recurrent_LBP, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.part_num = part_num
        if 'axu_loss' in kwargs:
            aux_loss = kwargs['axu_loss']
            self.compute_aux_loss = eval(aux_loss.pop('type'))(**aux_loss)
        else:
            self.compute_aux_loss = None

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # classifier
        for i in range(2):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(
                5120, self.num_classes, relu=True,
                dropout=False, bottle_dim=256)
                    )
        # setattr(self, 'classifier_g', BottleClassifier(
        #         2048, self.num_classes, relu=True,
        #         dropout=False, bottle_dim=256)
        #             )
                
        #self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.fc = nn.Linear(512, self.num_classes)
        self.fc_r = nn.Linear(2048**2, self.num_classes)
        self.fc_p_c = nn.Linear(self.in_channels*5, self.num_classes)
        self.fc_p_r = nn.Linear(self.in_channels*5, self.num_classes)

        #self.w = nn.Parameter(torch.ones(part_num)/part_num)
        self.w = nn.Parameter(torch.ones(4)/4)

        #### mamba ######
        self.config_g = MambaConfig(d_model=1, n_layers=1)
        self.mamba_model_g = Mamba(self.config_g)

        self.config_p = MambaConfig(d_model=1, n_layers=1)
        self.mamba_model_p = Mamba(self.config_p)


    def loss(self, score_list, gt_label, **kwargs):
            losses = dict()
            # compute loss

            globe_logits_list, cls_logits = score_list
            part_logits_list, avg_logits = cls_logits

            num_samples = len(gt_label)

            losses[f'loss_global'] = self.compute_loss(
                avg_logits, gt_label, avg_factor=num_samples, **kwargs)
            loss_all = losses[f'loss_global']
            
            for i in range(len(part_logits_list)):
                logits_i = part_logits_list[i]
                ide_loss_i = self.compute_loss(
                    logits_i, gt_label, avg_factor=self.part_num, **kwargs)
                losses[f'loss_{i}'] = 1.0 / float(len(part_logits_list)) * ide_loss_i
                # losses[f'loss_{i}'] = ide_loss_i
                loss_all += losses[f'loss_{i}']

            preds = []
            for i in range(len(gt_label)):
                pred = [logit[i][gt_label[i]] for logit in globe_logits_list]
                preds.append(pred)
            losses[f'new_apn_loss'] = pairwise_ranking_loss(preds)
            loss_all += losses[f'new_apn_loss']

            losses['loss'] = loss_all
            return losses

    def pre_logits(self, features):

        features_part_c, features_part_r, ini_features, recurrent_features = features
        feature_list = [ini_features]

        if features_part_c.dim() ==2:
            features_part_c=features_part_c.unsqueeze(0)
        for i in range(self.part_num):
            if self.part_num == 1:
                features_c = features_part_c
            else:
                if i==0:
                    features_c = features_part_c[:,:, i, :]
                else:
                    #features_ci= features_part_c[:,:, i, :]
                    features_c = torch.cat([features_c,features_part_c[:,:, i, :]], dim = 1)
        feature_list.append(features_c)

        if features_part_r.dim() ==3:
            features_part_r=features_part_r.unsqueeze(0)
        for j in range(self.part_num):
            if self.part_num == 1:
                features_r = features_part_r
            else:
                if j==0:
                    features_r = features_part_r[:, :, :, j]
                else:
                    #features_ri = features_part_r[:, :, :, j]
                    features_r = torch.cat([features_r,features_part_r[:, :, :, j]], dim = 1)        
        feature_list.append(features_r)
        feature_list.append(recurrent_features)

        return feature_list


    def get_logits(self, feature_list):
        avg_logits = 0
        part_logits_list = []
        globe_logits_list = []
    
        ini_logits = self.fc(feature_list[0])
        globe_logits_list.append(ini_logits)
        
        classifier_0 = getattr(self, 'classifier0')
        part_logits_c = classifier_0(feature_list[1].squeeze(2))
        part_logits_list.append(part_logits_c)

        classifier_1 = getattr(self, 'classifier1')
        part_logits_r = classifier_1(feature_list[2].squeeze(2))
        part_logits_list.append(part_logits_r)

        recurrent_logits = self.fc(feature_list[3])
        globe_logits_list.append(recurrent_logits)

        avg_logits = self.w[0] * part_logits_c + self.w[1] * part_logits_r + self.w[2] * ini_logits + self.w[3] * recurrent_logits
        return globe_logits_list, part_logits_list, avg_logits


    def forward_train(self, x, gt_label):

        feature_list = self.pre_logits(x)
        
        globe_logits_list, part_logits_list, avg_logits = self.get_logits(feature_list)
        losses = self.loss([globe_logits_list, (part_logits_list, avg_logits)], gt_label)

        return losses


    def simple_test(self, features, softmax=True, post_process=False):

        feature_list = self.pre_logits(features)
        globe_logits_list, part_logits_list, avg_logits = self.get_logits(feature_list)

        # cls_score = (global_logits + avg_logits) / 2
        # cls_score = 0.2 * global_logits + 0.8 * avg_logits
        cls_score = avg_logits
        #cls_score = globe_logits_list[1]

        if softmax:     #测试阶段绝对使用了softmax，即softmax=True
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        return pred




