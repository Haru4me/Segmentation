import torch
import torch.nn as nn


"""
    mean Intersection over Union metric
"""


class IoU(nn.Module):

    def __init__(self):

        super(IoU, self).__init__()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):

        smooth = 1e-8

        outputs = outputs.round().byte()
        labels = labels.round().byte()

        intersection = (outputs & labels).float().sum((2,3)).mean(0)
        union = (outputs | labels).float().sum((2,3)).mean(0)

        iou = (intersection + smooth) / (union + smooth)

        return iou



class DiceCoef(nn.Module):

    def __init__(self):

        super(DiceCoef, self).__init__()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):

        smooth = 1e-8

        outputs = outputs.round().byte()
        labels = labels.round().byte()

        numerator = 2 * (outputs & labels).float().sum((2,3)).mean(0)
        denominator = (outputs + labels).float().sum((2,3)).mean(0)

        return ((numerator + smooth) / (denominator + smooth))



class Accurasy(nn.Module):

    def __init__(self):

        super(Accurasy, self).__init__()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):

        outputs = outputs.round().byte()
        labels = labels.round().byte()

        numerator = (outputs == labels).float().sum((2,3)).mean(0)
        denominator = torch.ones(labels.shape).sum((2,3)).mean(0)

        acc = numerator / denominator

        return acc
