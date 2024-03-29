# import torch.nn as nn
# import torch
# from torchmetrics import JaccardIndex

# class AttentionGate(nn.Module):
#     def __init__(self, g_in_c, x_in_c):
#         super(AttentionGate, self).__init__()

#         self.g_conv_layer = nn.Conv2d(g_in_c, x_in_c, 1, 1)
#         self.x_conv_layer = nn.Conv2d(x_in_c, x_in_c, 1, 2)
#         self.si_conv_layer = nn.Conv2d(x_in_c*2, 1, 1, 1)
#         self.resampling = nn.Upsample(scale_factor=2)

#     def forward(self, g, x):
#         g = self.g_conv_layer(g)
#         g = torch.cat([g, self.x_conv_layer(x)], dim=1)
#         g = nn.ReLU()(g)
#         g = self.si_conv_layer(g)
#         g = nn.Sigmoid()(g)
#         g = self.resampling(g)
#         x = x*g
#         return x

# class ConvLayers(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(ConvLayers, self).__init__()

#         self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
#         self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
#         self.conv3 = nn.Conv2d(out_c + in_c, out_c, 3, padding=1)


#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.conv2(y)
#         y = torch.cat([y, x], dim=1)
#         y = self.conv3(y)
#         return nn.ReLU()(y)

# class DownSampling(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(DownSampling, self).__init__()
#         self.conv1 = ConvLayers(in_c=in_c, out_c=out_c)
#         self.conv2 = ConvLayers(in_c=out_c, out_c=out_c)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x, nn.MaxPool2d(2)(x)

# class UpSampling(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(UpSampling, self).__init__()
#         self.attention_layer = AttentionGate(in_c, out_c)
#         self.upsampling_layer = nn.Upsample(scale_factor=2)
#         self.conv_layer = ConvLayers(in_c + out_c, out_c)

#     def forward(self, x, intermediate_value):
#         intermediate_value = self.attention_layer(x, intermediate_value)
#         x = self.upsampling_layer(x)
#         x = torch.cat([x, intermediate_value], dim=1)
#         return self.conv_layer(x)



# class UNET(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(UNET, self).__init__()

#         self.layer1 = DownSampling(in_c, 32)
#         self.downLayers = nn.ModuleList([DownSampling(2**i, 2**(i + 1)) for i in range(5, 7)])
#         self.intermediate_layer = ConvLayers(2**(7), 2**(8))
#         self.upLayers = nn.ModuleList([UpSampling(2**i, 2**(i -1)) for i in range(8, 5, -1)])
#         self.final_layer = nn.Conv2d(32, out_channels=out_c, kernel_size=1)

#     def forward(self, x):
#         intermediate_values = []
#         i, x = self.layer1(x)
#         intermediate_values.append(i)
#         for layer in self.downLayers:
#             i, x = layer(x)
#             intermediate_values.append(i)
#         x = self.intermediate_layer(x)

#         for layer, i in zip(self.upLayers, intermediate_values[::-1]):
#             x = layer(x, i)

#         x = self.final_layer(x)
#         return x

# class SegmentationLoss(nn.Module):
#     def __init__(self, num_classes):
#         super(SegmentationLoss, self).__init__()
#         self.cross_entropy = torch.nn.CrossEntropyLoss()
#         self.iou = JaccardIndex(task="multiclass", num_classes=num_classes)

#     def forward(self, pred, ground_truth):
#         iou_loss = 1 - self.iou(pred, ground_truth)
#         cross_loss = self.cross_entropy(pred, ground_truth)

#         return iou_loss + cross_loss

# # model = UNET(3, 7)
# # input = torch.rand((1, 3, 512, 256))
# # print(model(input).shape)

import torch.nn as nn
import torch
import torch.nn.functional as F

# from torchmetrics import JaccardIndex

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
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.conv3 = nn.Conv2d(out_c + in_c, out_c, 3, padding=1)


    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = torch.cat([y, x], dim=1)
        y = self.conv3(y)
        return nn.ReLU()(y)

class DownSampling(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSampling, self).__init__()
        self.conv1 = ConvLayers(in_c=in_c, out_c=out_c)
        self.conv2 = ConvLayers(in_c=out_c, out_c=out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, nn.MaxPool2d(2)(x)

class UpSampling(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSampling, self).__init__()
        self.attention_layer = AttentionGate(in_c, out_c)
        self.upsampling_layer = nn.Upsample(scale_factor=2)
        self.conv_layer = ConvLayers(in_c + out_c, out_c)

    def forward(self, x, intermediate_value):
        intermediate_value = self.attention_layer(x, intermediate_value)
        x = self.upsampling_layer(x)
        x = torch.cat([x, intermediate_value], dim=1)
        return self.conv_layer(x)



class UNET(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNET, self).__init__()

        self.layer1 = DownSampling(in_c, 32)
        self.downLayers = nn.ModuleList([DownSampling(2**i, 2**(i + 1)) for i in range(5, 7)])
        self.intermediate_layer = ConvLayers(2**(7), 2**(8))
        self.upLayers = nn.ModuleList([UpSampling(2**i, 2**(i -1)) for i in range(8, 5, -1)])
        self.final_layer = nn.Conv2d(32, out_channels=out_c, kernel_size=1)

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
        return x


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)


        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, device):
        super(SegmentationLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.iou = IoULoss().to(device)

    def forward(self, pred, ground_truth):
        iou_loss = self.iou(pred, ground_truth)
        cross_loss = self.cross_entropy(pred, ground_truth)

        return cross_loss + iou_loss