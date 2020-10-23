import torch
import torch.nn as nn


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5):

        super(TverskyLoss, self).__init__()

        assert alpha + beta == 1

        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):

        smooth = 1e-8

        outputs = outputs
        labels = labels

        numerator = (outputs * labels).sum()
        denominator = (outputs * labels).sum() + self.alpha * (outputs * (1 - labels)).sum() +\
                        self.beta * ((1 - outputs) * labels).sum()

        return 1 - (numerator + smooth) / (denominator + smooth)
