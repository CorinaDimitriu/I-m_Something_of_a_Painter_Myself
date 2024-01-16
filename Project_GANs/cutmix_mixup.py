import numpy as np
import torch


def random_boundingbox(size, lam):
    width, height = size, size

    r = np.sqrt(1. - lam)
    w = int(width * r)
    h = int(height * r)
    x = np.random.randint(width)
    y = np.random.randint(height)

    x1 = np.clip(x - w // 2, 0, width)
    y1 = np.clip(y - h // 2, 0, height)
    x2 = np.clip(x + w // 2, 0, width)
    y2 = np.clip(y + h // 2, 0, height)

    return x1, y1, x2, y2


def CutMix(imsize):
    lam = np.random.beta(1, 1)
    x1, y1, x2, y2 = random_boundingbox(imsize, lam)
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (imsize * imsize))
    map = torch.ones((imsize, imsize))
    map[x1:x2, y1:y2] = 0
    if torch.rand(1) > 0.5:
        map = 1 - map
        lam = 1 - lam
    # lam is equivalent to map.mean()
    return map, x1, x2, y1, y2
