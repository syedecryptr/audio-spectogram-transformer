from train_config import *
import torch
import torch.nn as nn
import torch.nn.functional as F


def polyloss_criterion(out, label, epsilon=2.0):
    # print(out.shape, label.shape)
    ce = nn.CrossEntropyLoss()(out, label)
    pt = torch.mean(F.one_hot(label, NO_OF_CLASSES) * torch.nn.Softmax(dim=1)(
        out))  # perturbation loss
    return ce + epsilon * (1 - pt)


if __name__ == '__main__':
    out = torch.rand(64, 10)
    label = torch.rand(64).to(torch.long)
    print(polyloss_criterion(out, label))
