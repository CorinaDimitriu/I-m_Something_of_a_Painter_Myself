import glob
import os

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from torchvision.utils import make_grid


def build_dataset(root):
    files = sorted(glob.glob(os.path.join(root, "*.jpg")))
    return files, len(files)


def plot_images(t_images, lines, title="", detach=True):
    if detach:
        t_images = v2.Resize((64, 64), antialias=True)(t_images)
        t_images = t_images.detach().cpu() * 0.5 + 0.5
        grid = make_grid(t_images, nrow=lines).permute(1, 2, 0)
    else:
        t_images = torch.from_numpy(t_images)
        t_images = t_images.float() * 0.5 + 0.5
        grid = make_grid(t_images, nrow=lines)
    plt.figure(figsize=(10, 7))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.show()


def get_index_from_list(vals, time, x_shape):
    batch_size = time.shape[0]
    out = vals.gather(-1, time.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time.device)
