import torch
import torch.nn as nn
from piqa import SSIM

class AttentionGate(nn.Module):
    def __init__(self, g_in_c, x_in_c):
        super(AttentionGate, self).__init__()

        self.g_conv_layer = nn.Conv2d(g_in_c, x_in_c, 1, 1)
        self.x_conv_layer = nn.Conv2d(x_in_c, x_in_c, 1, 2)
        self.si_conv_layer = nn.Conv2d(x_in_c*2, 1, 1, 1)
        self.resampling = nn.Upsample(scale_factor=2)

    def forward(self, g, x):
        g = self.g_conv_layer(g)
        g = torch.cat([g, self.x_conv_layer(x)], dim=1)
        g = nn.ReLU()(g)
        g = self.si_conv_layer(g)
        g = nn.Sigmoid()(g)
        g = self.resampling(g)
        x = x*g
        return x

class ConvLayers(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c + in_c, out_c, 3, padding=1)
        self.batchNorm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        y = self.conv1(x)
        y = torch.cat([y, x], dim=1)
        y = self.conv2(y)
        y = self.batchNorm(y)
        return nn.ReLU()(y)

class DownSampling(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSampling, self).__init__()
        self.conv1 = ConvLayers(in_c=in_c, out_c=out_c)
        self.conv2 = ConvLayers(in_c=out_c, out_c=out_c)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.dropout(nn.MaxPool2d(2)(x))

class UpSampling(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSampling, self).__init__()
        self.attention_layer = AttentionGate(in_c, out_c)
        self.upsampling_layer = nn.Upsample(scale_factor=2)
        self.conv_layer = ConvLayers(in_c + out_c, out_c)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x, intermediate_value):
        intermediate_value = self.attention_layer(x, intermediate_value)
        x = self.upsampling_layer(x)
        x = torch.cat([x, intermediate_value], dim=1)
        return self.dropout(self.conv_layer(x))

class UNET(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNET, self).__init__()
        self.layer1 = DownSampling(in_c, 32)
        self.downLayers = nn.ModuleList([DownSampling(2**i, 2**(i + 1)) for i in range(5, 8)])
        self.intermediate_layer = ConvLayers(2**(8), 2**(9))
        self.upLayers = nn.ModuleList([UpSampling(2**i, 2**(i -1)) for i in range(9, 5, -1)])
        self.final_layer = nn.Conv2d(32, out_channels=out_c, kernel_size=1)
        self.activation_layer = nn.Sigmoid()

    def forward(self, x):
        intermediate_values = []
        i, x = self.layer1(x)
        intermediate_values.append(i)
        for layer in self.downLayers:
            i, x = layer(x)
            intermediate_values.append(i)
        x = self.intermediate_layer(x)

        for layer, i in zip(self.upLayers, intermediate_values[::-1]):
            x = layer(x, i)

        x = self.final_layer(x)
        return self.activation_layer(x)


class DepthEstimationLoss(nn.Module):
    def __init__(self):
        super(DepthEstimationLoss, self).__init__()
        self.mse_loss_layer = torch.nn.MSELoss()
        self.smooth_l1_loss_layer = torch.nn.SmoothL1Loss()
        self.ssim = SSIM(n_channels=1)

    def forward(self, predicted_depth, ground_truth_depth):
        MSE_loss = self.mse_loss_layer(predicted_depth, ground_truth_depth)
        smooth_l1_loss = self.smooth_l1_loss_layer(predicted_depth, ground_truth_depth)
        ssim_loss = (1. - self.ssim(predicted_depth, ground_truth_depth))/2
        return MSE_loss + smooth_l1_loss + ssim_loss

# model = UNET(3, 7)
# print(model)
# input = torch.rand((1, 3, 512, 256))
# print(model(input).shape)

# import matplotlib.pyplot as plt
# from torchvision.transforms import ToTensor

# lossFunction = DepthEstimationLoss()
# img = ToTensor()(plt.imread("D:/Major_Project_Initial/depth/test.png"))
# loss = lossFunction(img, img)
# print(loss)