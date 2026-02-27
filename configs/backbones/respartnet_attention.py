import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from .RecurrentNet import *
import timm

class Attention_ResPartNet(nn.Module):

    def __init__(self, part_num):
        super(Attention_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        # ##### self attention mechinism ######
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(1)

        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)


    def forward(self, x):

        #### convert RGB image into gray image
        # 定义灰度转换的权重系数
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

        # 将RGB图像转换为灰度图像
        x_gray = torch.sum(x * weights, dim=1, keepdim=True)
        #print("Gray image size:", x_gray.size())  # 应该输出 [32, 1, 512, 512]
        #x = x_gray.squeeze(1)     # x 为（32,512,512）对应（batch_size, seq_length, embedding_dim）
        
        x_gray = self.bn1(x_gray)
        print("attention input size:", x_gray)

        ##### self attention mechinism ######
        x = self.TokenAttention(x_gray)
        x = self.relu(self.bn0(self.conv0(x)))
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class ResPartNet_Tokken_Space_Attention(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet_Tokken_Space_Attention, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        # ##### self attention mechinism ######
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)
        


    def forward(self, x):

        #### convert RGB image into gray image
        # 定义灰度转换的权重系数
        #weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

        # 将RGB图像转换为灰度图像
        #x_gray = torch.sum(x * weights, dim=1, keepdim=True)  # 应该输出 [32, 1, 512, 512]
        #print("attention input size:", x.size())

        #####attention mechinism ######
        x = self.SpatialAttention(x)
        x = self.TokenAttention(x)
        x = self.relu(self.bn0(self.conv0(x)))
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Self_Attention_ResPartNet(nn.Module):

    def __init__(self, part_num):
        super(Self_Attention_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        # ##### self attention mechinism ######
        self.self_attention = SimplifiedScaledDotProductAttention(d_model=224, h=8)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        # self.SpatialAttention = SpatialAttention(kernel_size=7)
        # self.TokenAttention = TokenAttention(in_channels=16, ratio=16)       

    def forward(self, x):

        ### convert RGB image into gray image
        #定义灰度转换的权重系数
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to('cuda:0')

        #将RGB图像转换为灰度图像
        x_gray = torch.sum(x * weights, dim=1, keepdim=True)  # 应该输出 [32, 1, 512, 512]
        #print("attention input size:", x.size())
        x = x_gray.squeeze(1)

        #####attention mechinism ######
        x = self.self_attention(x,x,x)
        x = x.unsqueeze(1)
        #print("AAAA", x.size())
        x = self.relu(self.bn0(self.conv0(x)))
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class Tokken_ResPartNet(nn.Module):      

    '''''''''''
    Tokken_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Tokken_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)
        

    def forward(self, x):

        ##### Tokken_attention mechinism ######
        x = self.TokenAttention(x)
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class Recurrent_Tokken_ResPartNet(nn.Module):      

    '''''''''''
    Recurrent_Tokken_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Recurrent_Tokken_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        

        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
    
        ##### Tokken_attention mechinism ######
        x = self.TokenAttention(images)

        features = self.resnet_conv(x)
        print("features:",features.shape)
        attention_input = self.recurrentNet(features,images)
        print('attention_input',attention_input)
        recurrent_features = self.resnet_conv(attention_input)
        print('recurrent_features:',recurrent_features)
        return [features, recurrent_features]

class Recurrent_Tokken_ResPartNet_Dino_v2(nn.Module):

    """
    Recurrent_Token_Attention - DINOv2 backbone
    """

    def __init__(self, part_num, dino_name="dinov2_vitb14"):
        super().__init__()

        self.part_num = part_num

        # =========================
        # 1. DINOv2 backbone
        # =========================
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", dino_name, pretrained=True
        )
        self.backbone.eval()  # 通常 DINO 冻结更稳

        self.embed_dim = self.backbone.embed_dim  # 768 / 1024 / 1536

        self.TokenAttention = TokenAttention(
            in_channels=16, ratio=16)

        self.recurrentNet = RecurrentNet(num_classes=5)

    def forward(self, images):
        """
        images: [B, 3, H, W]
        """
        x = self.TokenAttention(images)

        # DINO feature extraction
        # output: [B, N+1, D] (CLS + patch tokens)
        tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
        B, N, D = tokens.shape
        H = W = int(D ** 0.5)
        # reshape tokens -> feature map
        features = tokens.transpose(1, 2).reshape(B, D, H, W)

        attention_input = self.recurrentNet(features, images)

        # 再过一次 DINO（可选：共享 or 冻结）
        tokens_r = self.backbone.forward_features(attention_input)["x_norm_patchtokens"]
        recurrent_features = tokens_r.transpose(1, 2).reshape(B, D, H, W)

        return [features, recurrent_features]


class ResPartNet_CBAM_Recurrent(nn.Module):      

    def __init__(self, part_num):
        super(ResPartNet_CBAM_Recurrent, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):

        x = self.resnet_conv(images)
        #####attention mechinism #####
        x = self.ChannelAttention(x)
        features = self.SpatialAttention(x)

        attention_input = self.recurrentNet(features,images)
        recurrent_features = self.resnet_conv(attention_input)
        
        return [features, recurrent_features]

class Tokken_ResPartNet_CBAM_Recurrent(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
        
        X = self.TokenAttention(images)
        features = self.resnet_conv(X)
        features = self.SpatialAttention(features)

        attention_input = self.recurrentNet(features,X)
        recurrent_features = self.resnet_conv(attention_input)
        
        return [features, recurrent_features]

class Tokken_ResPartNet_CBAM_Recurrent_v2(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
        
        X = self.TokenAttention(images)
        features = self.resnet_conv(X)
        print('features:',features.shape)
        features = self.ChannelAttention(features)
        features = self.SpatialAttention(features)

        attention_input = self.recurrentNet(features,X)
        print('attention_input:',attention_input.shape)
        recurrent_features = self.resnet_conv(attention_input)
        # recurrent_features = self.ChannelAttention(recurrent_features)
        # recurrent_features = self.SpatialAttention(recurrent_features)
        print('recurrent_features:',recurrent_features.shape)
        
        return [features, recurrent_features]

class Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v2_resnet50(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v2_resnet50, self).__init__()

        # attributes
        self.part_num = part_num

        #  =========================
        # 1. DINOv2 backbone
        # =========================
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True
        )
        # self.backbone = torch.hub.load(
        #     "/pretrain/dinov2",
        #     "dinov2_vitl14",
        #     source="local",
        #     pretrained=True
        # )
        #self.backbone.load_state_dict(torch.load("/public/home/2021002/data/jiyarong/MS_Project/pretrain/dinov2_vitl14_reg4_pretrain.pth"))
        #self.backbone.eval()  # 通常 DINO 冻结更稳
        # 首先：冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 其次：设置为训练模式
        self.backbone.train() 

        # 关键：只解冻最后 4 层 Blocks
        # ViT-L 通常有 24 层 (0-23)
        num_blocks = len(self.backbone.blocks)
        unfreeze_last_n = 4

        for i in range(num_blocks - unfreeze_last_n, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # 另外，通常建议解冻最后的 norm 层和 head 相关参数（如果有）
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        self.embed_dim = self.backbone.embed_dim  # 768 / 1024 / 1536

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.ChannelAttention = ChannelAttention(in_channels=1024, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
        
        X = self.TokenAttention(images)
        # DINO feature extraction
        # output: [B, N+1, D] (CLS + patch tokens)
        tokens = self.backbone.forward_features(X)["x_norm_patchtokens"]
        print('tokens:',tokens.shape)
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        # reshape tokens -> feature map
        features = tokens.transpose(1, 2).reshape(B, D, H, W)
        # features = self.resnet_conv(X)
        features = self.ChannelAttention(features)
        features = self.SpatialAttention(features)
        print('features:',features.shape)
        attention_input = self.recurrentNet(features,X)
        print('attention_input:',attention_input.shape)

        # tokens = self.backbone.forward_features(attention_input)["x_norm_patchtokens"]
        # B, N, D = tokens.shape
        # H = W = int(N ** 0.5)
        # # reshape tokens -> feature map
        # recurrent_features = tokens.transpose(1, 2).reshape(B, D, H, W)
        recurrent_features = self.resnet_conv(attention_input)
        
        return [features, recurrent_features]


class Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v2_ConvNeXt_V2(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v2_ConvNeXt_V2, self).__init__()
        #super().__init__()
        self.part_num = part_num

        # =========================
        # 1. DINOv2 backbone (frozen except last layers)
        # =========================
        # self.backbone_dino = torch.hub.load(
        #     "/pretrain/dinov2",
        #     "dinov2_vitl14",
        #     source="local",
        #     pretrained=True
        # )
        self.backbone_dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        for param in self.backbone_dino.parameters():
            param.requires_grad = False
        self.backbone_dino.train()  # keep in train mode for BN etc.

        num_blocks = len(self.backbone_dino.blocks)
        unfreeze_last_n = 4
        for i in range(num_blocks - unfreeze_last_n, num_blocks):
            for param in self.backbone_dino.blocks[i].parameters():
                param.requires_grad = True
        for param in self.backbone_dino.norm.parameters():
            param.requires_grad = True

        self.embed_dim_dino = self.backbone_dino.embed_dim  # 1024

        # =========================
        # 2. ConvNeXt-V2 backbone (frozen except last stage)
        # =========================
        #self.backbone_cnx = timm.create_model('convnextv2_base.fcmae', pretrained=True, features_only=True)
        # 替换原来的 backbone_cnx 初始化
        self.backbone_cnx = timm.create_model(
            'convnextv2_tiny.fcmae',
            pretrained=True,
            num_classes=0,      # 移除分类头
            global_pool=''      # 不做全局平均池化，保留空间维度
        )
        # features_only=True returns intermediate features, or use forward_features

        for param in self.backbone_cnx.parameters():
            param.requires_grad = False
        for param in self.backbone_cnx.stages[3].parameters():
            param.requires_grad = True
        # Optionally unfreeze final norm if exists
        if hasattr(self.backbone_cnx, 'norm_pre'):
            for param in self.backbone_cnx.norm_pre.parameters():
                param.requires_grad = True

        # =========================
        # 3. Attention modules
        # =========================
        self.ChannelAttention = ChannelAttention(in_channels=768, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes=5)

    def forward(self, images):
        # Apply token attention to input image
        X_att = self.TokenAttention(images)  # [B, 3, H, W]

        # DINO feature extraction
        tokens = self.backbone_dino.forward_features(X_att)["x_norm_patchtokens"]
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        dino_features = tokens.transpose(1, 2).reshape(B, D, H, W)
        dino_features = self.ChannelAttention(dino_features)
        dino_features = self.SpatialAttention(dino_features)

        # Optional: recurrent refinement (ensure output is [B, 3, H, W])
        refined_image = self.recurrentNet(dino_features, X_att)  # Must return image-like tensor

        # ConvNeXt feature extraction (from original or refined image)
        cnx_features = self.backbone_cnx(refined_image)  # or use images directly

        return [dino_features, cnx_features]

class Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v3_ConvNeXt_V2_vit(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v3_ConvNeXt_V2_vit, self).__init__()
        #super().__init__()
        self.part_num = part_num

        # 替换原来的 DINOv3 加载部分
        dinov3_repo_path = "/pretrain/dinov3"
        if dinov3_repo_path not in sys.path:
            sys.path.insert(0, os.path.dirname(dinov3_repo_path))  # 添加 pretrain/ 目录
        self.backbone_dino = torch.hub.load(
            "/pretrain/dinov3",
            "dinov3_vitb14_reg4",   # 注意：DINOv3 模型名通常带 _reg 后缀（表示使用 register token）
            source="local",
            pretrained=False        # 我们将手动加载权重
        )

        # 手动加载预训练权重
        ckpt_path = "/pretrain/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.backbone_dino.load_state_dict(state_dict, strict=True)

        # =========================
        # 2. ConvNeXt-V2 backbone (frozen except last stage)
        # =========================
        #self.backbone_cnx = timm.create_model('convnextv2_base.fcmae', pretrained=True, features_only=True)
        # 替换原来的 backbone_cnx 初始化
        self.backbone_cnx = timm.create_model(
            'convnextv2_base.fcmae',
            pretrained=True,
            num_classes=0,      # 移除分类头
            global_pool=''      # 不做全局平均池化，保留空间维度
        )
        # features_only=True returns intermediate features, or use forward_features

        for param in self.backbone_cnx.parameters():
            param.requires_grad = False
        for param in self.backbone_cnx.stages[3].parameters():
            param.requires_grad = True
        # Optionally unfreeze final norm if exists
        if hasattr(self.backbone_cnx, 'norm_pre'):
            for param in self.backbone_cnx.norm_pre.parameters():
                param.requires_grad = True

        # =========================
        # 3. Attention modules
        # =========================
        self.ChannelAttention = ChannelAttention(in_channels=768, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes=5)

    def forward(self, images):
        # Apply token attention to input image
        X_att = self.TokenAttention(images)  # [B, 3, H, W]

        # DINOv3 feature extraction
        with torch.no_grad():
            # 获取所有 tokens（包括 class token 和 register tokens）
            all_tokens = self.backbone_dino.forward_features(X_att)  # [B, N_total, D]
        
        # DINOv3 ViT-B/14 + reg4: total tokens = (H//14)*(W//14) + 1 (cls) + 4 (reg)
        # 我们只想要 patch tokens → 通常从索引 1 开始，跳过 cls 和 reg
        # 但注意：DINOv3 默认 **不输出 cls token**！只有 patch + register
        
        # 根据官方说明：register tokens are appended at the end
        # So: [patch_tokens (N), register_tokens (4)]
        num_register = 4
        patch_tokens = all_tokens[:, :-num_register]  # [B, N, D]

        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        dino_features = patch_tokens.transpose(1, 2).reshape(B, D, H, W)

        dino_features = self.ChannelAttention(dino_features)
        dino_features = self.SpatialAttention(dino_features)
        print('dino_features:',dino_features.shape)
        # Optional: recurrent refinement (ensure output is [B, 3, H, W])
        refined_image = self.recurrentNet(dino_features, X_att)  # Must return image-like tensor

        # ConvNeXt feature extraction (from original or refined image)
        cnx_features = self.backbone_cnx(refined_image)  # or use images directly

        return [dino_features, cnx_features]

class Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v3_ConvNeXt_V2_2(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v3_ConvNeXt_V2_2, self).__init__()
        #super().__init__()
        self.part_num = part_num

        # =========================
        # 1. DINOv3 ConvNeXt-Base 加载
        # =========================
        dino_repo_path = "/pretrain/dinov3"
        # 解决之前遇到的 ModuleNotFoundError
        if dino_repo_path not in sys.path:
            sys.path.insert(0, dino_repo_path)

        # 注意：这里根据 dinov3/hubconf.py 里的定义，
        # ConvNeXt Base 的名称通常是 "dinov3_convnext_base"
        try:
            self.backbone_dino = torch.hub.load(
                repo_or_dir=dino_repo_path,
                model="dinov3_convnext_base", 
                source="local",
                pretrained=False
            )
        except RuntimeError:
            # 如果报错，说明名字可能不对，建议检查 torch.hub.list(dino_repo_path, source="local")
            print("错误：请检查 hubconf.py 中 ConvNeXt Base 的确切名称")
            raise

        # 加载本地权重
        ckpt_path = "/pretrain/dinov3_convnext_base.pth"
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # 兼容处理：有些权重放在 'model' 键下
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.backbone_dino.load_state_dict(state_dict, strict=True)
        # for name, m in self.backbone_dino.named_modules():
        #     print(name)
        # self.backbone_dino.global_pool = torch.nn.Identity()
        # 冻结与微调策略 (ConvNeXt 结构与 ViT 不同，没有 blocks 属性)
        for param in self.backbone_dino.parameters():
            param.requires_grad = False
        
        # ConvNeXt 通常由 stages 组成，解冻最后一个 stage (stage 3)
        # 假设 DINOv3 的 ConvNeXt 保持了常规命名结构
        #print( self.backbone_dino.stages[3])
        if hasattr(self.backbone_dino, 'stages'):
            for param in self.backbone_dino.stages[2].parameters():
                param.requires_grad = True
            for param in self.backbone_dino.stages[3].parameters():
                param.requires_grad = True
        
        self.backbone_dino.train()

        # ConvNeXt-Base 的输出维度通常是 1024
        self.embed_dim_dino = 1024

        # =========================
        # 2. ConvNeXt-V2 backbone (frozen except last stage)
        # =========================
        #self.backbone_cnx = timm.create_model('convnextv2_base.fcmae', pretrained=True, features_only=True)
        # 替换原来的 backbone_cnx 初始化
        self.backbone_cnx = timm.create_model(
            'convnextv2_base.fcmae',
            pretrained=True,
            num_classes=0,      # 移除分类头
            global_pool=''      # 不做全局平均池化，保留空间维度
        )
        # features_only=True returns intermediate features, or use forward_features

        for param in self.backbone_cnx.parameters():
            param.requires_grad = False
        for param in self.backbone_cnx.stages[2].parameters():
            param.requires_grad = True
        for param in self.backbone_cnx.stages[3].parameters():
            param.requires_grad = True

        # Optionally unfreeze final norm if exists
        if hasattr(self.backbone_cnx, 'norm_pre'):
            for param in self.backbone_cnx.norm_pre.parameters():
                param.requires_grad = True

        # =========================
        # 3. Attention modules
        # =========================
        self.ChannelAttention = ChannelAttention(in_channels=1024, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes=5)

    def forward(self, images):
        # Apply token attention to input image
        X_att = self.TokenAttention(images)  # [B, 3, H, W]

        dino_features = self.backbone_dino.forward_features(X_att)["x_norm_patchtokens"]
        B, N, C = dino_features.shape
        H = W = int(N ** 0.5)   # 7
        dino_features = dino_features.transpose(1, 2).reshape(B, C, H, W)
        
        # if isinstance(dino_features, (list, tuple)):
        #     dino_features = dino_features[-1]
        #     print('DDD')
        
        dino_features = self.ChannelAttention(dino_features)
        dino_features = self.SpatialAttention(dino_features)
        
        # Optional: recurrent refinement (ensure output is [B, 3, H, W])
        refined_image = self.recurrentNet(dino_features, X_att)  # Must return image-like tensor

        # ConvNeXt feature extraction (from original or refined image)
        cnx_features = self.backbone_cnx(refined_image)  # or use images directly

        return [dino_features, cnx_features]

class Tokken_ResPartNet_dinov3(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_dinov3, self).__init__()

        # =========================
        # 1. DINOv3 ConvNeXt-Base 加载
        # =========================
        dino_repo_path = "/pretrain/dinov3"
        # 解决之前遇到的 ModuleNotFoundError
        if dino_repo_path not in sys.path:
            sys.path.insert(0, dino_repo_path)

        # 注意：这里根据 dinov3/hubconf.py 里的定义，
        # ConvNeXt Base 的名称通常是 "dinov3_convnext_base"
        try:
            self.backbone_dino = torch.hub.load(
                repo_or_dir=dino_repo_path,
                model="dinov3_convnext_base", 
                source="local",
                pretrained=False
            )
        except RuntimeError:
            # 如果报错，说明名字可能不对，建议检查 torch.hub.list(dino_repo_path, source="local")
            print("错误：请检查 hubconf.py 中 ConvNeXt Base 的确切名称")
            raise

        # 加载本地权重
        ckpt_path = "/pretrain/dinov3_convnext_base.pth"
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # 兼容处理：有些权重放在 'model' 键下
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.backbone_dino.load_state_dict(state_dict, strict=True)
    
        for param in self.backbone_dino.parameters():
            param.requires_grad = False
        
        
        if hasattr(self.backbone_dino, 'stages'):
            for param in self.backbone_dino.stages[2].parameters():
                param.requires_grad = True
            for param in self.backbone_dino.stages[3].parameters():
                param.requires_grad = True
        
        self.backbone_dino.train()

        # ConvNeXt-Base 的输出维度通常是 1024
        self.embed_dim_dino = 1024

        # =========================
        # 2. ConvNeXt-V2 backbone (frozen except last stage)
        # =========================
        #self.backbone_cnx = timm.create_model('convnextv2_base.fcmae', pretrained=True, features_only=True)
        # 替换原来的 backbone_cnx 初始化
        self.backbone_cnx = timm.create_model(
            'convnextv2_tiny.fcmae',
            pretrained=True,
            num_classes=0,      # 移除分类头
            global_pool=''      # 不做全局平均池化，保留空间维度
        )
        # features_only=True returns intermediate features, or use forward_features

        for param in self.backbone_cnx.parameters():
            param.requires_grad = False
        for param in self.backbone_cnx.stages[2].parameters():
            param.requires_grad = True
        for param in self.backbone_cnx.stages[3].parameters():
            param.requires_grad = True

        # Optionally unfreeze final norm if exists
        if hasattr(self.backbone_cnx, 'norm_pre'):
            for param in self.backbone_cnx.norm_pre.parameters():
                param.requires_grad = True

    def forward(self, images):
        
        dino_features = self.backbone_dino.forward_features(images)["x_norm_patchtokens"]
        B, N, C = dino_features.shape
        H = W = int(N ** 0.5)   # 7
        features = dino_features.transpose(1, 2).reshape(B, C, H, W)

        #cnx_features = self.backbone_cnx(refined_image)  # or use images directly

        return features

class Tokken_ResPartNet_convNext(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_convNext, self).__init__()

        # =========================
        # 2. ConvNeXt-V2 backbone (frozen except last stage)
        # =========================
        #self.backbone_cnx = timm.create_model('convnextv2_base.fcmae', pretrained=True, features_only=True)
        # 替换原来的 backbone_cnx 初始化
        self.backbone_cnx = timm.create_model(
            'convnextv2_tiny.fcmae',
            pretrained=True,
            num_classes=0,      # 移除分类头
            global_pool=''      # 不做全局平均池化，保留空间维度
        )
        # features_only=True returns intermediate features, or use forward_features

        for param in self.backbone_cnx.parameters():
            param.requires_grad = False
        for param in self.backbone_cnx.stages[2].parameters():
            param.requires_grad = True
        for param in self.backbone_cnx.stages[3].parameters():
            param.requires_grad = True

        # Optionally unfreeze final norm if exists
        if hasattr(self.backbone_cnx, 'norm_pre'):
            for param in self.backbone_cnx.norm_pre.parameters():
                param.requires_grad = True


    def forward(self, images):

        cnx_features = self.backbone_cnx(images)  # or use images directly

        return features


class Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v3(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2_dino_v3, self).__init__()

        # attributes
        self.part_num = part_num

        # 加载 DINOv3 系列中的 ConvNeXt (可选 tiny, small, base, large)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov3", "dinov3_convnext_base", pretrained=True)

        # 冻结参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ConvNeXt 结构不同，它分 stages。通常解冻最后的 stage 4
        # 对于 ConvNeXt-B，stages 索引通常是 3
        for param in self.backbone.stages[3].parameters():
            param.requires_grad = True
            
        # 别忘了最后的 norm 层
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.ChannelAttention = ChannelAttention(in_channels=1024, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
        
        X = self.TokenAttention(images)
        # DINO feature extraction
        # output: [B, N+1, D] (CLS + patch tokens)
        tokens = self.backbone.forward_features(X)["x_norm_patchtokens"]
        print('tokens:',tokens.shape)
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        # reshape tokens -> feature map
        features = tokens.transpose(1, 2).reshape(B, D, H, W)
        # features = self.resnet_conv(X)
        features = self.ChannelAttention(features)
        features = self.SpatialAttention(features)
        print('features:',features.shape)
        attention_input = self.recurrentNet(features,X)
        print('attention_input:',attention_input.shape)

        tokens = self.backbone.forward_features(attention_input)["x_norm_patchtokens"]
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        # reshape tokens -> feature map
        recurrent_features = tokens.transpose(1, 2).reshape(B, D, H, W)
        #recurrent_features = self.resnet_conv(attention_input)
        
        return [features, recurrent_features]


class Spectrum_ResPartNet(nn.Module):      

    '''''''''''
    Spectrum_Attention(ChannelAttention) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Spectrum_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        

        self.ChannelAttention = ChannelAttention(in_channels=224, ratio=16)
        

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ChannelAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class CBAM_ResPartNet(nn.Module):      

    '''''''''''
    CBAM_Attention(ChannelAttention-spatialAttention) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(CBAM_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        

        self.ChannelAttention = ChannelAttention(in_channels=224, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ChannelAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)

        x = self.SpatialAttention(x)
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class spatial_Spectrum_ResPartNet(nn.Module):      

    '''''''''''
    spatial_Spectrum_Attention(ChannelAttention) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(spatial_Spectrum_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.ChannelAttention = ChannelAttention(in_channels=224, ratio=16)
        

    def forward(self, x):

        x = self.SpatialAttention(x)
        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ChannelAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Spectrum_ECA_ResPartNet(nn.Module):      

    '''''''''''
    Spectrum_ECA_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Spectrum_ECA_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        

        self.ECAAttention = ECAAttention(kernel_size=3)
        

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ECAAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
       
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Spectrum_SE_ResPartNet(nn.Module):      

    '''''''''''
    Spectrum_SE_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Spectrum_SE_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.SEAttention = SEAttention(channel=224,reduction=8)
        

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.SEAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
       
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class ResPartNet_CBAM(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet_CBAM, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        
    def forward(self, x):

        x = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))

        #####attention mechinism #####
        x = self.ChannelAttention(x)
        features = self.SpatialAttention(x)
      
        return features

class ResPartNet_CA(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet_CA, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        # ##### self attention mechinism ######
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        
        self.CA = CA(inp=2048, reduction=16)
       


    def forward(self, x):

        x = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))

        #####attention mechinism #####
        features = self.CA(x)
      
        return features

class TSA_ResPartNet_CBAM(nn.Module):      

    '''''''''''
    Tokken_Space_Attention(TSA) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(TSA_ResPartNet_CBAM, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        #### TSA attention ######
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)

        ### CBAM attention #####
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

    def forward(self, x):

        ##### TSA_attention mechinism ######
        x = self.SpatialAttention(x)
        x = self.TokenAttention(x)
        #print("resnet input size:", x.size())

        x = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))

        #### CBAM attention #####
        x = self.ChannelAttention(x)
        features = self.SpatialAttention(x)

        return features

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x

class TokenAttention(nn.Module):
    """
    token注意力机制
    """

    def __init__(self, in_channels, ratio=16):
        super(TokenAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=3, keepdim=True) # （b,3,h,1）
        max_out, _ = torch.max(x, dim=3, keepdim=True)  
        out = avg_out + max_out
        #out = self.fc(out)
        out = self.sigmoid(out)
        return out * x

class TokenAttention_1(nn.Module):
    """
    token注意力机制
    """

    def __init__(self, in_channels, ratio=16):
        super(TokenAttention_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(torch.mean(x, dim=3, keepdim=True))   #(32,3,224,1)
        max_out, _ = self.fc(torch.max(x, dim=3, keepdim=True))  
        out = avg_out + max_out   #(32,3,224,2) 
        out = self.sigmoid(out) 
        return out * x


class CA(nn.Module):
    def __init__(self, inp, reduction):
        super(CA, self).__init__()
        # h:height(行)   w:width(列)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
 
        mip = max(8, inp // reduction)  # 论文作者所用
        # mip =  inp // reduction  # 博主所用   reduction = int(math.sqrt(inp))
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6












