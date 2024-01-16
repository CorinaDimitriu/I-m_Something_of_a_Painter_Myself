from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import torch

from cutmix_mixup import CutMix

if __name__ == '__main__':
    freeze_support()
    x = [torch.randn(1, 256, 256)] * 32
