import math

import pytorch_lightning as L
import torch
from diffusers import get_cosine_schedule_with_warmup
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from configs import DATALOADER_CONFIG
from cycleGAN import Upsampling, Downsampling
from utils import plot_images, get_index_from_list


class StableUpsamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = torch.nn.Sequential(torch.nn.Linear(time_emb_dim, out_channels),
                                            torch.nn.ReLU(True))
        self.conv1 = Downsampling(in_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.transform = Upsampling(out_channels, out_channels,
                                    kernel_size=4, stride=2, padding=1,
                                    relu=False, norm=False)
        self.conv2 = Downsampling(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        x = self.conv1(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.conv2(x)
        return self.transform(x)


class StableDownsamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = torch.nn.Sequential(torch.nn.Linear(time_emb_dim, out_channels),
                                            torch.nn.ReLU(True))
        self.conv1 = Downsampling(in_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.transform = Downsampling(out_channels, out_channels,
                                      kernel_size=4, stride=2, padding=1,
                                      relu=False, norm=False)
        self.conv2 = Downsampling(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        x = self.conv1(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.conv2(x)
        return self.transform(x)


class AttentionDownsamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, x_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = torch.nn.Sequential(torch.nn.Linear(time_emb_dim, out_channels),
                                            torch.nn.ReLU(True))
        self.conv1 = Downsampling(in_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.transform_wg = Downsampling(out_channels, out_channels,
                                         kernel_size=3, stride=1, padding=1,
                                         relu=False)
        self.transform_wx = Downsampling(x_channels, out_channels,
                                         kernel_size=4, stride=2, padding=1,
                                         relu=False)
        self.transform_psi = torch.nn.Sequential(
            Downsampling(out_channels, out_channels,
                         kernel_size=3, stride=1, padding=1,
                         relu=False),
            torch.nn.Sigmoid())
        self.conv2 = Downsampling(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, g, x, t):
        g = self.conv1(g)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        g = g + time_emb
        g = self.conv2(g)
        g1 = self.transform_wg(g)
        x1 = self.transform_wx(x)
        psi = torch.nn.ReLU()(g1 + x1)
        psi = self.transform_psi(psi)
        return x1 * psi


class AttentionUpsamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, x_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = torch.nn.Sequential(torch.nn.Linear(time_emb_dim, out_channels),
                                            torch.nn.ReLU(True))
        self.conv1 = Downsampling(in_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.transform_wg = Upsampling(out_channels, out_channels,
                                       kernel_size=3, stride=1, padding=1,
                                       relu=False)
        self.transform_wx = Upsampling(x_channels, out_channels,
                                       kernel_size=4, stride=2, padding=1,
                                       relu=False)
        self.transform_psi = torch.nn.Sequential(
            Upsampling(out_channels, out_channels,
                       kernel_size=3, stride=1, padding=1,
                       relu=False),
            torch.nn.Sigmoid())
        self.conv2 = Downsampling(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, g, x, t):
        g = self.conv1(g)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        g = g + time_emb
        g = self.conv2(g)
        g1 = self.transform_wg(g)
        x1 = self.transform_wx(x)
        psi = torch.nn.ReLU()(g1 + x1)
        psi = self.transform_psi(psi)
        return x1 * psi


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class StableUNetGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config['hidden_channels']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.time_emb_dim = config['time_embedding']

        # Time embedding
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
            torch.nn.ReLU(True)
        )

        # Initial projection
        self.conv0 = torch.nn.Conv2d(self.in_channels, self.hidden_channels, 3, padding=1)

        self.down_block = torch.nn.Sequential(StableDownsamplingBlock(self.hidden_channels,
                                                                      self.hidden_channels * 2,
                                                                      self.time_emb_dim),
                                              StableDownsamplingBlock(self.hidden_channels * 2,
                                                                      self.hidden_channels * 4,
                                                                      self.time_emb_dim),
                                              StableDownsamplingBlock(self.hidden_channels * 4,
                                                                      self.hidden_channels * 8,
                                                                      self.time_emb_dim),
                                              StableDownsamplingBlock(self.hidden_channels * 8,
                                                                      self.hidden_channels * 8,
                                                                      self.time_emb_dim),
                                              # StableDownsamplingBlock(self.hidden_channels * 8,
                                              #                         self.hidden_channels * 8,
                                              #                         self.time_emb_dim),
                                              AttentionDownsamplingBlock(self.hidden_channels * 8,
                                                                         self.hidden_channels * 8,
                                                                         self.hidden_channels * 8,
                                                                         self.time_emb_dim),
                                              StableDownsamplingBlock(self.hidden_channels * 8,
                                                                      self.hidden_channels * 8,
                                                                      self.time_emb_dim),
                                              )
        self.up_block = torch.nn.Sequential(StableUpsamplingBlock(self.hidden_channels * 8,
                                                                  self.hidden_channels * 8,
                                                                  self.time_emb_dim),
                                            AttentionUpsamplingBlock(self.hidden_channels * 16,
                                                                     self.hidden_channels * 8,
                                                                     self.hidden_channels * 8,
                                                                     self.time_emb_dim),
                                            # StableUpsamplingBlock(self.hidden_channels * 16,
                                            #                       self.hidden_channels * 8,
                                            #                       self.time_emb_dim),
                                            StableUpsamplingBlock(self.hidden_channels * 16,
                                                                  self.hidden_channels * 8,
                                                                  self.time_emb_dim),
                                            StableUpsamplingBlock(self.hidden_channels * 16,
                                                                  self.hidden_channels * 4,
                                                                  self.time_emb_dim),
                                            StableUpsamplingBlock(self.hidden_channels * 8,
                                                                  self.hidden_channels * 2,
                                                                  self.time_emb_dim),
                                            )

        self.feature_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(self.hidden_channels * 4, self.out_channels,
                                                                          kernel_size=4, stride=2, padding=1),
                                                 )

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        skips = []
        xPrev = Tensor(x)
        for down in self.down_block:
            if isinstance(down, AttentionDownsamplingBlock):
                x = down(x, xPrev, t)
                xPrev = Tensor(x)
            else:
                xPrev = Tensor(x)
                x = down(x, t)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_block, skips):
            if isinstance(up, AttentionUpsamplingBlock):
                x = up(x, xPrev, t)
                xPrev = Tensor(x)
            else:
                xPrev = Tensor(x)
                x = up(x, t)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)


class NoiseScheduler(torch.nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
        self.scheduler = torch.linspace(0.0001, 0.02, self.time_steps)

        alphas = 1. - self.scheduler
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        alphas_cum_prod_prev = torch.nn.functional.pad(alphas_cum_prod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cum_prod = torch.sqrt(alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - alphas_cum_prod)
        self.posterior_variance = self.scheduler * (1. - alphas_cum_prod_prev) / (1. - alphas_cum_prod)

    def forward_diffusion_sample(self, x_0, t, device=torch.device("cuda"), noise=True, sampled_noise=None):
        if noise:
            sampled_noise = torch.randn_like(x_0)
        sqrt_alphas_cum_prod_t = get_index_from_list(self.sqrt_alphas_cum_prod, t, x_0.shape)
        sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cum_prod, t, x_0.shape)
        # mean + variance
        if noise:
            return (torch.clamp(sqrt_alphas_cum_prod_t.to(device) * x_0.to(device)
                                + sqrt_one_minus_alphas_cum_prod_t.to(device) * sampled_noise.to(device), -1, 1),
                    sampled_noise.to(device))
        else:
            return torch.clamp(sqrt_alphas_cum_prod_t.to(device) * x_0.to(device)
                               + sqrt_one_minus_alphas_cum_prod_t.to(device) * sampled_noise.to(device), -1, 1)


# class NoiseScheduler(torch.nn.Module):
#     def __init__(self, time_steps):
#         super().__init__()
#         self.time_steps = time_steps
#         self.scheduler = torch.linspace(0.0001, 0.02, self.time_steps)
#
#         alphas = 1. - self.scheduler
#         alphas_cum_prod = torch.cumprod(alphas, dim=0)
#         self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
#         alphas_cum_prod_prev = torch.nn.functional.pad(alphas_cum_prod[:-1], (1, 0), value=1.0)
#         self.sqrt_alphas_cum_prod = torch.sqrt(alphas_cum_prod)
#         self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - alphas_cum_prod)
#         self.posterior_variance = self.scheduler * (1. - alphas_cum_prod_prev) / (1. - alphas_cum_prod)
#
#     def forward_diffusion_sample(self, x_0, t, device=torch.device("cuda"), noise=True):
#         if noise:
#             noise = torch.randn_like(x_0)
#         sqrt_alphas_cum_prod_t = get_index_from_list(self.sqrt_alphas_cum_prod, t, x_0.shape)
#         sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
#             self.sqrt_one_minus_alphas_cum_prod, t, x_0.shape)
#         # mean + variance
#         if noise:
#             return (torch.clamp(sqrt_alphas_cum_prod_t.to(device) * x_0.to(device)
#                                 + sqrt_one_minus_alphas_cum_prod_t.to(device) * noise.to(device), -1, 1),
#                     noise.to(device))
#         else:
#             return torch.clamp(sqrt_alphas_cum_prod_t.to(device) * x_0.to(device)
#                                + sqrt_one_minus_alphas_cum_prod_t.to(device) * noise.to(device), -1, 1)


class StableDiffuser(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config['lr']
        self.betas = config['betas']
        self.optimizer = config['optimizer']
        self.num_epochs = config['num_epochs']
        self.lr_warmup_steps = config['decay_epochs']
        self.automatic_optimization = False
        self.mean_init = config['mean_weight_init']
        self.std_init = config['std_weight_init']
        self.model = StableUNetGenerator(config)
        self.noise_scheduler = NoiseScheduler(config['time_steps'])
        self.device1 = config['device']

    def forward(self, x):
        for time in range(self.noise_scheduler.time_steps - 1, -1, -1):
            t = torch.Tensor([time]).type(torch.int64).to(self.device1)
            betas_t = get_index_from_list(self.noise_scheduler.scheduler, t, x.shape)
            sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
                self.noise_scheduler.sqrt_one_minus_alphas_cum_prod, t, x.shape)
            sqrt_recip_alphas_t = get_index_from_list(self.noise_scheduler.sqrt_recip_alphas, t, x.shape)

            # Call model (current image - noise prediction)
            model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cum_prod_t)
            if time == 0:
                x = model_mean
            else:
                noise = torch.randn_like(x)
                posterior_variance_t = get_index_from_list(self.noise_scheduler.posterior_variance, t, x.shape)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
        return torch.clamp(x, -1, 1)

    def setup(self, stage):
        def init_fn(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d)):
                torch.nn.init.normal_(m.weight, self.mean_init, self.std_init)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        if stage == "fit":
            for net in self.model.modules():
                net.apply(init_fn)
            print("Model initialized.")

    def get_lr_scheduler(self, optimizer):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * (DATALOADER_CONFIG['len_dataset']
                                                     / DATALOADER_CONFIG['batch_size']),
            num_training_steps=self.num_epochs * (DATALOADER_CONFIG['len_dataset']
                                                  / DATALOADER_CONFIG['batch_size']),
        )
        return lr_scheduler

    def configure_optimizers(self):
        opt_config = {
            "lr": self.lr,
            "betas": self.betas,
        }
        opt_gen = self.optimizer(list(self.model.parameters()), **opt_config)
        optimizers = [opt_gen]
        schedulers = []
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        real_M = batch
        bs = real_M.shape[0]

        time_steps = torch.randint(
            0, self.noise_scheduler.time_steps, (bs,), device=real_M.device
        ).long()
        noisy_images, noise = self.noise_scheduler.forward_diffusion_sample(real_M, time_steps)
        noise_pred = self.model(noisy_images, time_steps)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        opt = self.optimizers()
        # scheduler = self.lr_schedulers()
        self.toggle_optimizer(opt)
        opt.zero_grad()
        self.manual_backward(loss)
        # clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        # scheduler.step()
        self.untoggle_optimizer(opt)

        metrics = {
            "gen_loss": loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_index):
        self.display_results(batch)

    def predict_step(self, batch, batch_index):
        return self(torch.randn_like(batch))

    def display_results(self, batch):
        real_P = torch.randn_like(batch)
        fake_M = self(real_P)

        title = f"Sample {self.current_epoch + 1}: Photo-to-Monet Translation"

        plot_images(fake_M, lines=len(fake_M), title=title)
