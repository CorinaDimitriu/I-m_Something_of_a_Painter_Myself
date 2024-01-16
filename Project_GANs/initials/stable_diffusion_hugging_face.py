import numpy as np
import pytorch_lightning as L
import torch
from diffusers import DDPMScheduler, get_cosine_schedule_with_warmup, DDPMPipeline
from diffusers import UNet2DModel
from torch.nn.utils import clip_grad_norm_

from configs import DATALOADER_CONFIG
from utils import plot_images


class StableDiffuser_HuggingFace(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        hidden_channels = config['hidden_channels']
        self.model = UNet2DModel(
            sample_size=DATALOADER_CONFIG['image_size'],
            in_channels=config['in_channels'],  # the number of input channels, 3 for RGB images
            out_channels=config['out_channels'],  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(hidden_channels * 1,
                                hidden_channels * 2,
                                hidden_channels * 4,
                                hidden_channels * 8,
                                hidden_channels * 8,
                                hidden_channels * 8,
                                ),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.lr = config['lr']
        self.betas = config['betas']
        self.optimizer = config['optimizer']
        self.num_epochs = config['num_epochs']
        self.lr_warmup_steps = config['decay_epochs']
        self.sample_size = DATALOADER_CONFIG['sample_size']
        self.seed = config['seed']
        self.automatic_optimization = False
        self.mean_init = config['mean_weight_init']
        self.std_init = config['std_weight_init']

    def setup(self, stage):
        def init_fn(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d)):
                torch.nn.init.normal_(m.weight, self.mean_init, self.std_init)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        if stage == "fit":
            for net in [self.model]:
                net.apply(init_fn)
            print("Model initialized.")

    def forward(self, x, batch_size):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)
        return pipeline(
            batch_size=batch_size,
            generator=torch.manual_seed(self.seed),
            output_type=np.array
        ).images

    def get_lr_scheduler(self, optimizer):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_epochs * (DATALOADER_CONFIG['len_dataset'] / DATALOADER_CONFIG['batch_size']),
        )
        return lr_scheduler

    def configure_optimizers(self):
        opt_config = {
            "lr": self.lr,
            "betas": self.betas,
        }
        opt_gen = self.optimizer(list(self.model.parameters()), **opt_config)
        optimizers = [opt_gen]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        real_M = batch["monet"]
        noise = torch.randn(real_M.shape).to(real_M.device)
        bs = real_M.shape[0]

        time_steps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=real_M.device
        ).long()
        noisy_images = self.noise_scheduler.add_noise(real_M, noise, time_steps)
        noise_pred = self.model(noisy_images, time_steps, return_dict=False)[0]
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        self.toggle_optimizer(opt)
        self.model.zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        self.untoggle_optimizer(opt)

        metrics = {
            "gen_loss": loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_index):
        self.display_results(batch)

    def predict_step(self, batch, batch_index):
        return self(batch, len(batch))

    def display_results(self, batch):
        real_P = batch
        fake_M = self(real_P, len(batch))

        title = f"Sample {self.current_epoch + 1}: Photo-to-Monet Translation"

        plot_images(fake_M, lines=len(fake_M), title=title, detach=False)
