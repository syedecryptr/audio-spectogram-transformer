import torch
from torchvision.models import resnet50
from torch import nn


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet_model = resnet50(pretrained=True,
                                     replace_stride_with_dilation=[False,
                                                                   True,
                                                                   True])

        # replace the first layer with 1 channel image
        self.layer1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.rest_layers = nn.Sequential(
            *list(self.resnet_model.children())[1:9])
        self.linear = nn.Linear(in_features=2048, out_features=10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer1(x)
        x = self.rest_layers(x)
        x = x.reshape(x.shape[0], x.shape[1])
        x = self.linear(x)

        return x


if __name__ == '__main__':
    res = Resnet()
    print(res(torch.rand(2, 1, 51, 123)).shape)
