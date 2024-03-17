import torch
import torch.nn as nn
from torchsummary.torchsummary import summary


class _QueueLayer(nn.Sequential):
    def __init__(self, feature_in, fm, cr):
        super(_QueueLayer, self).__init__()
        self.add_module('conv1', nn.Conv3d(feature_in, int(fm * cr),
                                           kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False))
        self.add_module('conv2', nn.Conv3d(int(fm * cr), int(fm * cr),
                                           kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False))
        self.add_module('conv3', nn.Conv3d(int(fm * cr), int(fm * cr),
                                           kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False))
        self.add_module('conv4', nn.Conv3d(int(fm * cr), int(fm * cr),
                                           kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False))
        self.add_module('conv5', nn.Conv3d(int(fm * cr), fm,
                                           kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = super(_QueueLayer, self).forward(x)
        return out


class _QueneBlock(nn.Sequential):
    def __init__(self, block_id, feature_in, fm, cr):
        super(_QueneBlock, self).__init__()
        block = _QueueLayer(feature_in, fm, cr)
        self.add_module(f'block{block_id}', block)


class volume_net(nn.Module):
    def __init__(self, fm=32, cr=0.5):
        super().__init__()
        # First convolution
        self.F00 = nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=False)

        # Each queue block
        self.block1_1 = _QueneBlock(block_id='1_1', feature_in=8, fm=fm, cr=cr)
        self.block1_2 = _QueneBlock(block_id='1_2', feature_in=fm, fm=fm, cr=cr)
        self.block1_3 = _QueneBlock(block_id='1_3', feature_in=fm, fm=fm, cr=cr)
        self.block1_4 = _QueneBlock(block_id='1_4', feature_in=fm, fm=fm, cr=cr)
        self.block1_5 = _QueneBlock(block_id='1_5', feature_in=fm, fm=fm, cr=cr)
        self.block1_6 = _QueneBlock(block_id='1_6', feature_in=fm, fm=fm, cr=cr)
        self.block1_7 = _QueneBlock(block_id='1_7', feature_in=fm, fm=fm, cr=cr)
        self.block1_8 = _QueneBlock(block_id='1_8', feature_in=fm, fm=fm, cr=cr)
        self.block1_9 = _QueneBlock(block_id='1_9', feature_in=fm, fm=fm, cr=cr)
        self.block1_10 = _QueneBlock(block_id='1_10', feature_in=fm, fm=fm, cr=cr)
        self.block1_11 = _QueneBlock(block_id='1_11', feature_in=fm, fm=fm, cr=cr)
        self.block1_12 = _QueneBlock(block_id='1_12', feature_in=fm, fm=fm, cr=cr)

        self.block2_1 = _QueneBlock(block_id='2_1', feature_in=fm, fm=fm, cr=cr)
        self.block2_2 = _QueneBlock(block_id='2_2', feature_in=fm, fm=fm, cr=cr)
        self.block2_3 = _QueneBlock(block_id='2_3', feature_in=fm, fm=fm, cr=cr)
        self.block2_4 = _QueneBlock(block_id='2_4', feature_in=fm, fm=fm, cr=cr)
        self.block2_5 = _QueneBlock(block_id='2_5', feature_in=fm, fm=fm, cr=cr)
        self.block2_6 = _QueneBlock(block_id='2_6', feature_in=fm, fm=fm, cr=cr)
        self.block2_7 = _QueneBlock(block_id='2_7', feature_in=fm, fm=fm, cr=cr)
        self.block2_8 = _QueneBlock(block_id='2_8', feature_in=fm, fm=fm, cr=cr)
        self.block2_9 = _QueneBlock(block_id='2_9', feature_in=fm, fm=fm, cr=cr)
        self.block2_10 = _QueneBlock(block_id='2_10', feature_in=fm, fm=fm, cr=cr)
        self.block2_11 = _QueneBlock(block_id='2_11', feature_in=fm, fm=fm, cr=cr)

        self.block3_1 = _QueneBlock(block_id='3_1', feature_in=2*fm, fm=fm, cr=cr)
        self.block3_2 = _QueneBlock(block_id='3_2', feature_in=fm, fm=fm, cr=cr)
        self.block3_3 = _QueneBlock(block_id='3_3', feature_in=fm, fm=fm, cr=cr)
        self.block3_4 = _QueneBlock(block_id='3_4', feature_in=fm, fm=fm, cr=cr)
        self.block3_5 = _QueneBlock(block_id='3_5', feature_in=fm, fm=fm, cr=cr)
        self.block3_6 = _QueneBlock(block_id='3_6', feature_in=fm, fm=fm, cr=cr)
        self.block3_7 = _QueneBlock(block_id='3_7', feature_in=fm, fm=fm, cr=cr)
        self.block3_8 = _QueneBlock(block_id='3_8', feature_in=fm, fm=fm, cr=cr)
        self.block3_9 = _QueneBlock(block_id='3_9', feature_in=fm, fm=fm, cr=cr)
        self.block3_10 = _QueneBlock(block_id='3_10', feature_in=fm, fm=fm, cr=cr)

        self.block4_1 = _QueneBlock(block_id='4_1', feature_in=3*fm, fm=fm, cr=cr)
        self.block4_2 = _QueneBlock(block_id='4_2', feature_in=fm, fm=fm, cr=cr)
        self.block4_3 = _QueneBlock(block_id='4_3', feature_in=fm, fm=fm, cr=cr)
        self.block4_4 = _QueneBlock(block_id='4_4', feature_in=fm, fm=fm, cr=cr)
        self.block4_5 = _QueneBlock(block_id='4_5', feature_in=fm, fm=fm, cr=cr)
        self.block4_6 = _QueneBlock(block_id='4_6', feature_in=fm, fm=fm, cr=cr)
        self.block4_7 = _QueneBlock(block_id='4_7', feature_in=fm, fm=fm, cr=cr)
        self.block4_8 = _QueneBlock(block_id='4_8', feature_in=fm, fm=fm, cr=cr)
        self.block4_9 = _QueneBlock(block_id='4_9', feature_in=fm, fm=fm, cr=cr)

        self.block5_1 = _QueneBlock(block_id='5_1', feature_in=4*fm, fm=fm, cr=cr)
        self.block5_2 = _QueneBlock(block_id='5_2', feature_in=fm, fm=fm, cr=cr)
        self.block5_3 = _QueneBlock(block_id='5_3', feature_in=fm, fm=fm, cr=cr)
        self.block5_4 = _QueneBlock(block_id='5_4', feature_in=fm, fm=fm, cr=cr)
        self.block5_5 = _QueneBlock(block_id='5_5', feature_in=fm, fm=fm, cr=cr)
        self.block5_6 = _QueneBlock(block_id='5_6', feature_in=fm, fm=fm, cr=cr)
        self.block5_7 = _QueneBlock(block_id='5_7', feature_in=fm, fm=fm, cr=cr)
        self.block5_8 = _QueneBlock(block_id='5_8', feature_in=fm, fm=fm, cr=cr)

        self.block6_1 = _QueneBlock(block_id='6_1', feature_in=5*fm, fm=fm, cr=cr)
        self.block6_2 = _QueneBlock(block_id='6_2', feature_in=fm, fm=fm, cr=cr)
        self.block6_3 = _QueneBlock(block_id='6_3', feature_in=fm, fm=fm, cr=cr)
        self.block6_4 = _QueneBlock(block_id='6_4', feature_in=fm, fm=fm, cr=cr)
        self.block6_5 = _QueneBlock(block_id='6_5', feature_in=fm, fm=fm, cr=cr)
        self.block6_6 = _QueneBlock(block_id='6_6', feature_in=fm, fm=fm, cr=cr)
        self.block6_7 = _QueneBlock(block_id='6_7', feature_in=fm, fm=fm, cr=cr)

        self.block7_1 = _QueneBlock(block_id='7_1', feature_in=6*fm, fm=fm, cr=cr)
        self.block7_2 = _QueneBlock(block_id='7_2', feature_in=fm, fm=fm, cr=cr)
        self.block7_3 = _QueneBlock(block_id='7_3', feature_in=fm, fm=fm, cr=cr)
        self.block7_4 = _QueneBlock(block_id='7_4', feature_in=fm, fm=fm, cr=cr)
        self.block7_5 = _QueneBlock(block_id='7_5', feature_in=fm, fm=fm, cr=cr)
        self.block7_6 = _QueneBlock(block_id='7_6', feature_in=fm, fm=fm, cr=cr)

        self.block8_1 = _QueneBlock(block_id='8_1', feature_in=7*fm, fm=fm, cr=cr)
        self.block8_2 = _QueneBlock(block_id='8_2', feature_in=fm, fm=fm, cr=cr)
        self.block8_3 = _QueneBlock(block_id='8_3', feature_in=fm, fm=fm, cr=cr)
        self.block8_4 = _QueneBlock(block_id='8_4', feature_in=fm, fm=fm, cr=cr)
        self.block8_5 = _QueneBlock(block_id='8_5', feature_in=fm, fm=fm, cr=cr)

        self.block9_1 = _QueneBlock(block_id='9_1', feature_in=8*fm, fm=fm, cr=cr)
        self.block9_2 = _QueneBlock(block_id='9_2', feature_in=fm, fm=fm, cr=cr)
        self.block9_3 = _QueneBlock(block_id='9_3', feature_in=fm, fm=fm, cr=cr)
        self.block9_4 = _QueneBlock(block_id='9_4', feature_in=fm, fm=fm, cr=cr)

        self.block10_1 = _QueneBlock(block_id='10_1', feature_in=9*fm, fm=fm, cr=cr)
        self.block10_2 = _QueneBlock(block_id='10_2', feature_in=fm, fm=fm, cr=cr)
        self.block10_3 = _QueneBlock(block_id='10_3', feature_in=fm, fm=fm, cr=cr)

        self.block11_1 = _QueneBlock(block_id='11_1', feature_in=10*fm, fm=fm, cr=cr)
        self.block11_2 = _QueneBlock(block_id='11_2', feature_in=fm, fm=fm, cr=cr)

        self.block12_1 = _QueneBlock(block_id='12_1', feature_in=11*fm, fm=fm, cr=cr)

        self.allin = nn.Conv3d(12*fm, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.recon = nn.Conv3d(8, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        x_00 = self.F00(x)
        x_11 = self.block1_1(x_00)
        x_12 = self.block1_2(x_11)
        x_13 = self.block1_3(x_12)
        x_14 = self.block1_4(x_13)
        x_15 = self.block1_5(x_14)
        x_16 = self.block1_6(x_15)
        x_17 = self.block1_7(x_16)
        x_18 = self.block1_8(x_17)
        x_19 = self.block1_9(x_18)
        x_1_10 = self.block1_10(x_19)
        x_1_11 = self.block1_11(x_1_10)
        x_1_12 = self.block1_12(x_1_11)

        x_21 = self.block2_1(x_11)
        x_22 = self.block2_2(x_21)
        x_23 = self.block2_3(x_22)
        x_24 = self.block2_4(x_23)
        x_25 = self.block2_5(x_24)
        x_26 = self.block2_6(x_25)
        x_27 = self.block2_7(x_26)
        x_28 = self.block2_8(x_27)
        x_29 = self.block2_9(x_28)
        x_2_10 = self.block2_10(x_29)
        x_2_11 = self.block2_11(x_2_10)

        x_30 = torch.cat([x_12, x_21], 1)
        x_31 = self.block3_1(x_30)
        x_32 = self.block3_2(x_31)
        x_33 = self.block3_3(x_32)
        x_34 = self.block3_4(x_33)
        x_35 = self.block3_5(x_34)
        x_36 = self.block3_6(x_35)
        x_37 = self.block3_7(x_36)
        x_38 = self.block3_8(x_37)
        x_39 = self.block3_9(x_38)
        x_3_10 = self.block3_10(x_39)

        x_40 = torch.cat([x_13, x_22, x_31], 1)
        x_41 = self.block4_1(x_40)
        x_42 = self.block4_2(x_41)
        x_43 = self.block4_3(x_42)
        x_44 = self.block4_4(x_43)
        x_45 = self.block4_5(x_44)
        x_46 = self.block4_6(x_45)
        x_47 = self.block4_7(x_46)
        x_48 = self.block4_8(x_47)
        x_49 = self.block4_9(x_48)

        x_50 = torch.cat([x_14, x_23, x_32, x_41], 1)
        x_51 = self.block5_1(x_50)
        x_52 = self.block5_2(x_51)
        x_53 = self.block5_3(x_52)
        x_54 = self.block5_4(x_53)
        x_55 = self.block5_5(x_54)
        x_56 = self.block5_6(x_55)
        x_57 = self.block5_7(x_56)
        x_58 = self.block5_8(x_57)

        x_60 = torch.cat([x_15, x_24, x_33, x_42, x_51], 1)
        x_61 = self.block6_1(x_60)
        x_62 = self.block6_2(x_61)
        x_63 = self.block6_3(x_62)
        x_64 = self.block6_4(x_63)
        x_65 = self.block6_5(x_64)
        x_66 = self.block6_6(x_65)
        x_67 = self.block6_7(x_66)

        x_70 = torch.cat([x_16, x_25, x_34, x_43, x_52, x_61], 1)
        x_71 = self.block7_1(x_70)
        x_72 = self.block7_2(x_71)
        x_73 = self.block7_3(x_72)
        x_74 = self.block7_4(x_73)
        x_75 = self.block7_5(x_74)
        x_76 = self.block7_6(x_75)

        x_80 = torch.cat([x_17, x_26, x_35, x_44, x_53, x_62, x_71], 1)
        x_81 = self.block8_1(x_80)
        x_82 = self.block8_2(x_81)
        x_83 = self.block8_3(x_82)
        x_84 = self.block8_4(x_83)
        x_85 = self.block8_5(x_84)

        x_90 = torch.cat([x_18, x_27, x_36, x_45, x_54, x_63, x_72, x_81], 1)
        x_91 = self.block9_1(x_90)
        x_92 = self.block9_2(x_91)
        x_93 = self.block9_3(x_92)
        x_94 = self.block9_4(x_93)

        x_10_0 = torch.cat([x_19, x_28, x_37, x_46, x_55, x_64, x_73, x_82, x_91], 1)
        x_10_1 = self.block10_1(x_10_0)
        x_10_2 = self.block10_2(x_10_1)
        x_10_3 = self.block10_3(x_10_2)

        x_11_0 = torch.cat([x_1_10, x_29, x_38, x_47, x_56, x_65, x_74, x_83, x_92, x_10_1], 1)
        x_11_1 = self.block11_1(x_11_0)
        x_11_2 = self.block11_2(x_11_1)

        x_12_0 = torch.cat([x_1_11, x_2_10, x_39, x_48, x_57, x_66, x_75, x_84, x_93, x_10_2, x_11_1], 1)
        x_12_1 = self.block12_1(x_12_0)

        x_all_in = torch.cat([x_1_12, x_2_11, x_3_10, x_49, x_58, x_67, x_76, x_85, x_94, x_10_3, x_11_2, x_12_1], 1)

        return self.recon(self.allin(x_all_in) + x_00)

    def weight_init(self):
        for m in self._modules:
            kaiming_init(self._modules[m])


def kaiming_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight.data)


if __name__ == '__main__':
    # test code
    input_shape = (1, 16, 16, 16)
    model = volume_net()
    summary(model, input_shape)
