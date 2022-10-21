import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=78, output_ch=128):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=1, stride=1)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.dense1 = nn.Linear(2048, 1024)
        self.dense = nn.Linear(1024, 2)

    def calc_accuracy(self, Y_Pred: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Get the accuracy with respect to the most likely label

        :param Y_Pred:
        :param Y:
        :return:
        """

        # return the values & indices with the largest value in the dimension where the scores for each class is
        # get the scores with largest values & their corresponding idx (so the class that is most likely)
        max_scores, max_idx_class = Y_Pred.max(
            dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
        # usually 0th coordinate is batch size
        n = Y.size(0)
        assert (n == max_idx_class.size(0))
        # calulate acc (note .item() to do float division)
        acc = (max_idx_class == Y).sum().item() / n
        return acc

    def forward(self, x):
        x = torch.reshape(x, list(x.shape) + [-1])
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)

        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.reshape(d1, (x.shape[0], -1))
        d1 = F.sigmoid(self.dense1(d1))
        d1 = F.sigmoid(self.dense(d1))

        return d1


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv1d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                    stride=2, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class CNNCodeDuplExt(torch.nn.Module):
    def calc_accuracy(self, Y_Pred: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Get the accuracy with respect to the most likely label

        :param Y_Pred:
        :param Y:
        :return:
        """

        # return the values & indices with the largest value in the dimension where the scores for each class is
        # get the scores with largest values & their corresponding idx (so the class that is most likely)
        max_scores, max_idx_class = Y_Pred.max(
            dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
        # usually 0th coordinate is batch size
        n = Y.size(0)
        assert (n == max_idx_class.size(0))
        # calulate acc (note .item() to do float division)
        acc = (max_idx_class == Y).sum().item() / n
        return acc

    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(78, affine=False)
        self.conv = nn.Conv1d(78, 32, 1, stride=2)
        self.deconv = nn.ConvTranspose1d(32, 242, kernel_size=1)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.215)
        self.dense1 = nn.Linear(121, 80)
        self.dense = nn.Linear(80, 2)

    def forward(self, x):
        x = torch.reshape(x, list(x.shape) + [-1])
        x = self.batch_norm(x)
        x = self.relu(self.conv(x))
        x = self.relu(self.deconv(x))
        x = torch.reshape(x, (-1, 242))
        x = self.relu(self.maxpool(x))
        x = self.dropout(x)
        x = F.sigmoid(self.dense1(x))

        x = F.sigmoid(self.dense(x))
        return x


class CNNCodeDuplExtResUnet(torch.nn.Module):
    def calc_accuracy(self, Y_Pred: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Get the accuracy with respect to the most likely label

        :param Y_Pred:
        :param Y:
        :return:
        """

        # return the values & indices with the largest value in the dimension where the scores for each class is
        # get the scores with largest values & their corresponding idx (so the class that is most likely)
        max_scores, max_idx_class = Y_Pred.max(
            dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
        # usually 0th coordinate is batch size
        n = Y.size(0)
        assert (n == max_idx_class.size(0))
        # calulate acc (note .item() to do float division)
        acc = (max_idx_class == Y).sum().item() / n
        return acc

    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(78, affine=False)
        self.conv = nn.Conv1d(78, 32, 1, stride=2)
        self.deconv = nn.ConvTranspose1d(32, 242, kernel_size=1)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.215)
        self.dense1 = nn.Linear(121, 80)
        self.dense = nn.Linear(274, 2)
        self.conv1 = nn.Conv1d(274, 121, 1)
        self.attent = Attention_block(242, 32, 242)
        self.conv2 = nn.Conv1d(121, 121, 1)


    def forward(self, x):
        x = torch.reshape(x, list(x.shape) + [-1])
        x = self.batch_norm(x)
        x1 = self.relu(self.conv(x))
        x2 = self.relu(self.deconv(x1))
        # x1 = self.attent(x2,x1)
        x3 = torch.cat([x1, x2], dim=1)
        #
        # x4 = self.relu(self.maxpool(x4))
        # x4 = self.dropout(x4)
        # x4=self.relu(self.conv1(x3))
        # x4=self.relu(self.conv2(x4))
        x4 = torch.reshape(x3, (-1, 274))
        x4 = self.dropout(x4)
        # x4 = F.sigmoid(self.dense1(x4))

        x4 = F.sigmoid(self.dense(x4))
        return x4


class CNNCodeDuplExtResUnetAtt(torch.nn.Module):
    def calc_accuracy(self, Y_Pred: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Get the accuracy with respect to the most likely label

        :param Y_Pred:
        :param Y:
        :return:
        """

        # return the values & indices with the largest value in the dimension where the scores for each class is
        # get the scores with largest values & their corresponding idx (so the class that is most likely)
        max_scores, max_idx_class = Y_Pred.max(
            dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
        # usually 0th coordinate is batch size
        n = Y.size(0)
        assert (n == max_idx_class.size(0))
        # calulate acc (note .item() to do float division)
        acc = (max_idx_class == Y).sum().item() / n
        return acc

    def __init__(self):
        super().__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(78, affine=False)
        self.conv = nn.Conv1d(78, 32, 1)
        self.conv1 = nn.Conv1d(32, 16, 1)
        self.conv2 = nn.Conv1d(16, 8, 1)
        self.deconv1 = nn.ConvTranspose1d(8, 16, kernel_size=1)
        self.batch_norm1=nn.BatchNorm1d(16)
        self.deconv2 = nn.ConvTranspose1d(16, 32, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.deconv3 = nn.ConvTranspose1d(32, 80, kernel_size=1)
        self.deconv4 = nn.ConvTranspose1d(64, 240, kernel_size=1)
        self.batch_norm3 = nn.BatchNorm1d(80)
        self.attent1 = Attention_block(16, 8, 16)
        self.attent2 = Attention_block(32, 16, 32)
        self.attent3 = Attention_block(80, 32, 80)
        self.batch_norm = nn.BatchNorm1d(78, affine=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.215)
        self.dense1 = nn.Linear(240, 120)
        self.batch_normf1 = nn.BatchNorm1d(120, affine=False)
        self.dense2 = nn.Linear(120, 60)
        self.batch_normf2 = nn.BatchNorm1d(60, affine=False)
        self.dense3 = nn.Linear(60, 30)
        self.batch_normf3 = nn.BatchNorm1d(30, affine=False)
        self.dense4 = nn.Linear(30, 2)



    def forward(self, x):
        x = torch.reshape(x, list(x.shape) + [-1])
        x =  self.batch_norm(x)
        x1 = self.conv(x)
        x1 = self.Maxpool(x1)

        x2 = self.conv1(x1)
        x2 = self.Maxpool(x2)

        x3 = self.conv2(x2)
        x3 = self.Maxpool(x3)

# decoder start

        x4 = self.relu(self.deconv1(x3))
        x4 = self.batch_norm1(x4)
        x4 = self.attent1(x4, x3)
        x5 = torch.cat([x3, x4], dim=1)

        x6 = self.relu(self.deconv2(x5))
        x6 = self.batch_norm2(x6)
        x6 = self.attent2(x6, x2)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.relu(self.deconv3(x7))
        x8 = self.batch_norm3(x8)
        x8 = self.attent3(x8, x1)
        x9 = torch.cat([x8, x1], dim=1)
        x9 = self.deconv4(x9)
        x10 = torch.reshape(x9, (x9.shape[0],-1))
        x10 = self.relu(self.Maxpool(x10))

        x10 = self.dropout(x10)
        x10 = self.batch_normf1(F.sigmoid(self.dense1(x10)))
        x10 = self.batch_normf2(F.sigmoid(self.dense2(x10)))
        x10 = self.batch_normf3(F.sigmoid(self.dense3(x10)))
        x10 = F.sigmoid(self.dense4(x10))

        return x10
