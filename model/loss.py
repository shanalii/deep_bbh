import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def binary_cross_entropy(output, target):
	return F.binary_cross_entropy(output, target)