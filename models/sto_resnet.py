import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import StoConv2d, StoLinear, StoLayer, StoModel


# def sto_conv3x3(in_planes, out_planes, stride, groups, dilation, use_bnn, prior_mean, prior_std, posterior_mean_init, posterior_std_init, **kwargs):
#     """3x3 convolution with padding"""
#     return StoConv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#         use_bnn=use_bnn,
#         prior_mean=prior_mean,
#         prior_std=prior_std,
#         posterior_mean_init=posterior_mean_init,
#         posterior_std_init=posterior_std_init,
#         **kwargs
#     )

# def sto_conv1x1(in_planes, out_planes, stride, use_bnn, prior_mean, prior_std, posterior_mean_init, posterior_std_init, **kwargs):
#     """1x1 convolution"""
#     return StoConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding=0, dilation=1, groups=1, \
#                      use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, **kwargs)




class StoBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, \
                 use_bnn=False, prior_mean=0, prior_std=1, posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), same_noise=False, sigma_parameterization='softplus'):
        super(StoBasicBlock, self).__init__()
        self.conv1 = StoConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, \
            use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
            sigma_parameterization=sigma_parameterization)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StoConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, \
                          use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                            sigma_parameterization=sigma_parameterization),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, \
                 use_bnn=False, prior_mean=0, prior_std=1, posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), same_noise=False, sigma_parameterization='softplus'):
        super(StoBottleneck, self).__init__()
        self.conv1 = StoConv2d(in_planes, planes, kernel_size=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = StoConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, \
                               use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                sigma_parameterization=sigma_parameterization)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                StoConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, \
                          use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                            sigma_parameterization=sigma_parameterization),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StoResNet(nn.Module, StoModel):
    def __init__(self, block, num_blocks, num_classes=10, \
                 use_bnn=False, prior_mean=0, prior_std=1, posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), same_noise=False, sigma_parameterization='softplus'):
        super(StoResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = StoConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                    sigma_parameterization=sigma_parameterization)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                        sigma_parameterization=sigma_parameterization)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                        sigma_parameterization=sigma_parameterization)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                        sigma_parameterization=sigma_parameterization)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, \
                                       use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                        sigma_parameterization=sigma_parameterization)
        self.linear = StoLinear(512*block.expansion, num_classes, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                    sigma_parameterization=sigma_parameterization)

    def _make_layer(self, block, planes, num_blocks, stride, \
                    use_bnn=False, prior_mean=0, prior_std=1, posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), same_noise=False, sigma_parameterization='softplus'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, \
                                use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                                    sigma_parameterization=sigma_parameterization))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_test_mean(self, test_with_mean):
        for m in self.modules():
            if isinstance(m, StoLayer):
                m.test_with_mean = test_with_mean

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def StoResNet18(num_classes, use_bnn=False, prior_mean=0, prior_std=1, posterior_mean_init=(0, 0.1), posterior_std_init=(0, 0.1), same_noise=False, sigma_parameterization='softplus'):
    return StoResNet(StoBasicBlock, [2, 2, 2, 2], num_classes=num_classes, \
                     use_bnn=use_bnn, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init, same_noise=same_noise, \
                        sigma_parameterization=sigma_parameterization)