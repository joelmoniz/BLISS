from __future__ import absolute_import
from torch import Tensor
from torch.autograd import Variable
from bliss.utils import to_cuda


class GaussianAdditive(object):
    def __init__(self, gpu, std):
        self.gpu = gpu
        self.std = std

    def generate(self, embed):
        # if self.gpu:
        #    _noise = Tensor(embed.size(0), embed.size(1)).normal_(mean=0, std=self.std)
        # else:
        #    _noise = torch.cuda.Tensor(embed.size(0), embed.size(1)).normal_(mean=0, std=self.std)
        _noise = to_cuda(Tensor(embed.size(0), embed.size(1)).normal_(mean=0, std=self.std), self.gpu)
        return Variable(embed.data + _noise)
