import torch
import torch.nn as nn
from torchsummary.torchsummary import summary


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features)),
        self.add_module('elu', nn.LeakyReLU(inplace=True)),
        self.add_module('conv', nn.Conv3d(num_input_features, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        # Concatenation
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Generator(nn.Module):
    """a copy of mDCSRN b4u4"""
    def __init__(self, block_config=(4, 4, 4, 4), growth_rate=16):

        super(Generator, self).__init__()
        # First convolution
        self.conv0 = nn.Conv3d(1, 2 * growth_rate, kernel_size=3, padding=1, bias=False)

        # Each denseblock
        num_features = 2 * growth_rate
        num_features_cat = num_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, growth_rate=growth_rate)
        num_features_cat += block_config[0] * growth_rate + num_features
        self.comp0 = nn.Conv3d(num_features_cat, num_features, kernel_size=1, stride=1, bias=False)

        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, growth_rate=growth_rate)
        num_features_cat += block_config[1] * growth_rate + num_features
        self.comp1 = nn.Conv3d(num_features_cat, num_features, kernel_size=1, stride=1, bias=False)

        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, growth_rate=growth_rate)
        num_features_cat += block_config[2] * growth_rate + num_features
        self.comp2 = nn.Conv3d(num_features_cat, num_features, kernel_size=1, stride=1, bias=False)

        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, growth_rate=growth_rate)
        num_features_cat += block_config[3] * growth_rate + num_features
        self.recon = nn.Conv3d(num_features_cat, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        out = self.block0(x)
        features = torch.cat([x, out], 1)
        out = self.comp0(features)

        out = self.block1(out)
        features = torch.cat([features, out], 1)
        out = self.comp1(features)

        out = self.block2(out)
        features = torch.cat([features, out], 1)
        out = self.comp2(features)

        out = self.block3(out)
        features = torch.cat([features, out], 1)
        out = self.recon(features)
        return out

    def weight_init(self):
        for m in self._modules:
            kaiming_init(self._modules[m])


class Discriminator(nn.Module):

    def __init__(self, cube_size=64):
        super(Discriminator, self).__init__()
        num_features = 64
        self.main = nn.Sequential(
            nn.Conv3d(1, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(1),

            nn.Conv3d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([num_features, cube_size // 2, cube_size // 2, cube_size // 2]),
            nn.LeakyReLU(1),

            nn.Conv3d(num_features, 2 * num_features, kernel_size=3, padding=1),
            nn.LayerNorm([2 * num_features, cube_size // 2, cube_size // 2, cube_size // 2]),
            nn.LeakyReLU(1),

            nn.Conv3d(2 * num_features, 2 * num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([2 * num_features, cube_size // 4, cube_size // 4, cube_size // 4]),
            nn.LeakyReLU(1),

            nn.Conv3d(2 * num_features, 4 * num_features, kernel_size=3, padding=1),
            nn.LayerNorm([4 * num_features, cube_size // 4, cube_size // 4, cube_size // 4]),
            nn.LeakyReLU(1),

            nn.Conv3d(4 * num_features, 4 * num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([4 * num_features, cube_size // 8, cube_size // 8, cube_size // 8]),
            nn.LeakyReLU(1),

            nn.Conv3d(4 * num_features, 8 * num_features, kernel_size=3, padding=1),
            nn.LayerNorm([8 * num_features, cube_size // 8, cube_size // 8, cube_size // 8]),
            nn.LeakyReLU(1),

            nn.Conv3d(8 * num_features, 8 * num_features, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([8 * num_features, cube_size // 16, cube_size // 16, cube_size // 16]),
            nn.LeakyReLU(1),

            # different from the original SRGAN, we replaced the FC layers by
            # global averaging pooling and convolution layers.
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(8 * num_features, 16 * num_features, kernel_size=1),
            nn.LeakyReLU(1),
            nn.Conv3d(16 * num_features, 1, kernel_size=1)
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(x.size()[0])

    def weight_init(self):
        for m in self._modules:
            kaiming_init(self._modules[m])


def kaiming_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight.data)


if __name__ == '__main__':
    # test code
    input_shape = (1, 32, 32, 32)
    model = Discriminator(cube_size=32)
    summary(model, input_shape)
