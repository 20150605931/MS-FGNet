import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.fft

class CompactBilinearPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.h = torch.randint(0, out_dim, (in_dim,))
        self.s = torch.randint(0, 2, (in_dim,)) * 2 - 1

    def sketch(self, x):
        # x: [B, D]
        B, D = x.size()
        sketch = x.new_zeros(B, self.out_dim)
        for i in range(D):
            sketch[:, self.h[i]] += self.s[i] * x[:, i]
        return sketch

    def forward(self, x):
        x_sketch = self.sketch(x)
        fft_x = torch.fft.fft(x_sketch)
        cbp = torch.real(torch.fft.ifft(fft_x * fft_x))
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-8)
        return nn.functional.normalize(cbp)

class LowRankBilinearPooling(nn.Module):
    def __init__(self, in_dim, out_dim, rank=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, rank, bias=False)
        self.fc2 = nn.Linear(in_dim, rank, bias=False)
        self.fc_out = nn.Linear(rank, out_dim)

    def forward(self, x):
        # x: [B, D]
        z1 = self.fc1(x)
        z2 = self.fc2(x)
        z = z1 * z2                  # Hadamard product
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)  # optional
        z = nn.functional.normalize(z)
        return self.fc_out(z)

class PartGlobalAveragePooling_recurrent_LBP(nn.Module):

    def __init__(self, part_num):
        super(PartGlobalAveragePooling_recurrent_LBP, self).__init__()
        self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        self.avgpool_r = nn.AdaptiveAvgPool2d((1, part_num))
        dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_c = nn.Sequential(self.avgpool_c, dropout)
        self.pool_r = nn.Sequential(self.avgpool_r, dropout)
        
        self.LBP = LowRankBilinearPooling(1024,512,768)
        #self.LBP_r = LowRankBilinearPooling(768,512,768)


    def forward(self, features):
        ini_features, recurrent_features = features
        features_part_c = self.pool_c(ini_features)
        features_part_r = self.pool_r(ini_features)
    
        ini_features = self.LBP(self.maxpool(ini_features).squeeze())
        recurrent_features = self.LBP(self.maxpool(recurrent_features).squeeze())
        # ini_features = self.maxpool(ini_features)
        # recurrent_features = self.maxpool(recurrent_features)
        return features_part_c, features_part_r, ini_features, recurrent_features

