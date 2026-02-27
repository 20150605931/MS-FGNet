import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from .vgg import vgg19_bn

class RecurrentNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(RecurrentNet, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False # make sure

        self.feature_pool1 = nn.AvgPool2d(kernel_size = 28, stride = 28)
        self.feature_pool2 = nn.AvgPool2d(kernel_size = 14, stride = 14)
        
        self.atten_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
       
        self.apn1 = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 3),
            nn.Sigmoid(),
        )

        self.apn2 = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 512),
            nn.Tanh(),
            nn.Linear(512, 3),
            nn.Sigmoid(),
        )
        self.crop_resize = AttentionCropLayer()


    def forward(self, features,x):

        atten1 = self.apn1(features.reshape(features.size(0), -1))
        scaledA_x = self.crop_resize(x, atten1 * 224)
        
        return scaledA_x

class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()
        
        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tl = tl if tl > (in_size/3) else in_size/3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk
            
            xatt_cropped = xatt[:, w_off:w_end, h_off:h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(112,112), mode='bilinear', align_corners = True)
            ret.append(xamp.data.squeeze())
        
        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 112
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size/3*2)
        short_size = (in_size/3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x<short_size)+(x>=long_size)+(y<short_size)+(y>=long_size)) > 0).float()*2 - 1
        
        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))
        
        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()
        
        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret

class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)
