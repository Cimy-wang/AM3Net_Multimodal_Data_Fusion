import torch.nn.functional as F
import torch
import torch.nn as nn
from involution import involution


class Feature_HSI(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Feature_HSI, self).__init__()
        self.stride = stride
        self.F_1 = nn.Conv2d(in_channels, 16, kernel_size, stride, padding, dilation, groups, bias)
        self.F_2 = nn.Conv2d(16, 32, kernel_size, stride, padding, dilation, groups, bias)
        self.F_3 = nn.Conv2d(32, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, X_h):
        X_h = self.F_3(self.F_2(self.F_1(X_h)))
        return X_h


class Spectral_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spectral_Weight, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.f_inv_12 = involution(in_channels, kernel_size, 1)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.relu(self.bn_h(self.f_inv_11(self.f_inv_12(X_h))))
        return X_h


class Spatial_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spatial_Weight, self).__init__()
        self.Conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.relu(self.bn_h(self.Conv_weight(X_h)))
        return X_h


class Spatial_feature(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spatial_feature, self).__init__()
        self.Conv_feature_1 = nn.Conv2d(in_channels, middle_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Conv_feature_2 = nn.Conv2d(middle_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Conv_feature_3 = nn.Conv2d(in_channels, middle_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Conv_feature_4 = nn.Conv2d(middle_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Spatial_weight = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.Spectral_weight = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_spectral, X_spatial):
        X_spectral_ = self.relu(self.bn_h(self.Conv_feature_2(self.Conv_feature_1(X_spectral)) * self.Spectral_weight(X_spatial)))
        X_spatial_ = self.relu(self.bn_h(self.Conv_feature_4(self.Conv_feature_3(X_spatial)) * self.Spatial_weight(X_spectral)))
        return X_spectral_, X_spatial_


class Dimension(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Dimension, self).__init__()
        self.Conv_feature_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_1):
        X_1 = self.relu(self.bn_h(self.Conv_feature_1(X_1)))
        return X_1


class Feature_Fusion(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Feature_Fusion, self).__init__()
        self.weight_lambda = nn.Parameter(torch.ones(3), requires_grad=True)
        self.Fusion_feature_1 = Dimension(64, out_channels, kernel_size, 2, padding, dilation, groups, bias)
        self.Fusion_feature_2 = Dimension(64, out_channels, kernel_size, 2, padding, dilation, groups, bias)
        self.Fusion_feature_3 = Dimension(128, out_channels, kernel_size, 1, padding, dilation, groups, bias)
        self.Fusion_feature_4 = Dimension(128, out_channels, kernel_size, 1, padding, dilation, groups, bias)
        self.Fusion_feature_5 = Dimension(256, out_channels, 1, stride, 0, dilation, groups, bias)
        self.Fusion_feature_6 = Dimension(256, out_channels, 1, stride, 0, dilation, groups, bias)

    def forward(self, X_1, X_2, X_3, X_4, X_5, X_6):
        fusion_feature_1 = self.Fusion_feature_1(X_1)
        fusion_feature_2 = self.Fusion_feature_2(X_2)
        fusion_feature_3 = self.Fusion_feature_3(X_3)
        fusion_feature_4 = self.Fusion_feature_4(X_4)
        cat_feature_1 = torch.cat((fusion_feature_1, fusion_feature_2), dim=1)
        cat_feature_2 = torch.cat((fusion_feature_3, fusion_feature_4), dim=1)
        cat_feature_3 = torch.cat((X_5, X_6), dim=1)
        weight = F.softmax(self.weight_lambda, 0)
        feature_final = weight[0] * cat_feature_1 + weight[1] * cat_feature_2 + weight[2] * cat_feature_3
        return feature_final


class octfusion_multi_adder_1(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super(octfusion_multi_adder_1, self).__init__()
        self.Weight_Alpha = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.Feature = nn.Conv2d(in_channels_1, 15, kernel_size=1, stride=1, padding=0)
        self.Feature_HSI = Feature_HSI(15, 64, kernel_size=1, stride=1, padding=0)
        self.Spectral_Weight = Spectral_Weight(15, 64, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_HSI = Spatial_Weight(15, 64, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_LIDAR = Spatial_Weight(in_channels_2, 64, kernel_size=3, stride=1, padding=1)
        self.Spatial_Spectral_Feature_1 = Spatial_feature(64, 96, 128, kernel_size=3, stride=1, padding=1)
        self.Spatial_Spectral_Feature_2 = Spatial_feature(128, 192, 256, kernel_size=3, stride=1, padding=1)
        self.Feature_Fusion = Feature_Fusion(256, kernel_size=3, stride=1, padding=1)
        self.GAP = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256*2, out_channels)

    def forward(self, x1, x2):
        in_size = x1.size(0)
        if x1.size(0) != 15:
            x1 = self.Feature(x1)
            feature_hsi = self.Feature_HSI(x1)
        else:
            feature_hsi = self.Feature_HSI(x1)
        spectral_weight = self.Spectral_Weight(x1)
        spatial_Weight_hsi = self.Spatial_Weight_HSI(x1)
        spatial_Weight_lidar = self.Spatial_Weight_LIDAR(x2)
        feature_spctral = spectral_weight * feature_hsi
        weight_alpha = F.softmax(self.Weight_Alpha)
        feature_spatial = (weight_alpha[0] * spatial_Weight_hsi + weight_alpha[1] * spatial_Weight_lidar) * feature_hsi
        feature_spctral = F.avg_pool2d(feature_spctral, (2, 2))
        feature_spatial = F.avg_pool2d(feature_spatial, (2, 2))
        feature_spctral_1, feature_spatial_1 = self.Spatial_Spectral_Feature_1(feature_spctral, feature_spatial)
        feature_spctral_1 = F.avg_pool2d(feature_spctral_1, (2, 2))
        feature_spatial_1 = F.avg_pool2d(feature_spatial_1, (2, 2))
        feature_spctral_2, feature_spatial_2 = self.Spatial_Spectral_Feature_2(feature_spctral_1, feature_spatial_1)
        feature_fusion = self.Feature_Fusion(feature_spctral, feature_spatial, feature_spctral_1, feature_spatial_1, feature_spctral_2, feature_spatial_2)
        out = self.GAP(feature_fusion)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.softmax(out, dim=1)
        return out

