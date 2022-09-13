import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNCodeDuplExt(torch.nn.Module):
    def calc_accuracy(self,Y_Pred:torch.Tensor,  Y: torch.Tensor) -> float:
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
        self.batch_norm= nn.BatchNorm1d(78, affine=False)
        self.conv = nn.Conv1d(78, 32, 1, stride=2)
        self.deconv = nn.ConvTranspose1d(32,242,kernel_size=1)
        self.maxpool =  nn.MaxPool1d(2, stride=2)
        self.dropout=nn.Dropout(p=0.215)
        self.dense1 = nn.Linear(121, 80)
        self.dense=nn.Linear(80, 2)

    def forward(self, x):
        x=torch.reshape(x, list(x.shape) + [-1])
        x=self.batch_norm(x)
        x=F.relu(self.conv(x))
        x=F.relu(self.deconv(x))
        x=torch.reshape(x,(-1,242))
        x=F.relu(self.maxpool(x))
        x=self.dropout(x)
        x = F.sigmoid(self.dense1(x))

        x=  F.sigmoid(self.dense(x))
        return x
