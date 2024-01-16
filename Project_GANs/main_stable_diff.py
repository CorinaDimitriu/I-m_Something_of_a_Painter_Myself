import os
import shutil
from multiprocessing import freeze_support

import pytorch_lightning as L
from torchvision.transforms import v2
from torchvision.utils import save_image

from configs import DATALOADER_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from unet_discriminator_cycleGAN import CycleGAN
from loader import CustomLoader
from processing import Augmentation
from stable_diffusion_new_GAN import DiffusionCycleGAN
from stable_diffusion_standalone import StableDiffuser

if __name__ == '__main__':
    freeze_support()

    dm = CustomLoader(config=DATALOADER_CONFIG, transform=Augmentation())
    model = DiffusionCycleGAN(config=MODEL_CONFIG)
    trainer = L.Trainer(**TRAIN_CONFIG)
    trainer.fit(model, datamodule=dm)
    # dm.setup("fit")
    # train_loader = dm.train_dataloader()
    # for i in range(2):
    #     image1 = (next(iter(train_loader))[0]['monet1'][0].unsqueeze(0))
    #     image2 = (next(iter(train_loader))[0]['monet2'][0].unsqueeze(0))
    # num_images = 10
    # stepsize = int(MODEL_CONFIG['time_steps'] / num_images)
    # images = []
    # for idx in range(0, MODEL_CONFIG['time_steps'], stepsize):
    #     t = torch.Tensor([idx]).type(torch.int64)
    #     noise_scheduler = NoiseScheduler(time_steps=MODEL_CONFIG['time_steps'])
    #     img, noise = noise_scheduler.forward_diffusion_sample(image, t)
    #     images.append(img)
    # plot_images(torch.cat(images, dim=0), num_images)

    predictions = trainer.predict(model, datamodule=dm)
    os.makedirs(".\\images", exist_ok=True)
    idx = 0
    for tensor in predictions:
        for monet in tensor:
            save_image(
                monet.float().squeeze() * 0.5 + 0.5,
                fp=f".\\images\\{idx}.jpg",
            )
            idx += 1

    shutil.make_archive(".\\images", "zip", ".\\images")
