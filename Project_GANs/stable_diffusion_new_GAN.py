import math

import numpy as np
import pytorch_lightning as L
import torch
from torch import Tensor
from torchvision.transforms import v2

from configs import DATALOADER_CONFIG
from cutmix_mixup import CutMix
from cycleGAN import Upsampling, Downsampling
from stable_diffusion_standalone import NoiseScheduler, AttentionUpsamplingBlock, AttentionDownsamplingBlock, \
    StableUpsamplingBlock, StableDownsamplingBlock, SinusoidalPositionEmbeddings
from utils import plot_images, get_index_from_list


class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_channels = config['hidden_channels']
        in_channels = config['in_channels']
        self.block = torch.nn.Sequential(
            Downsampling(in_channels, hidden_channels, norm=False),
            Downsampling(hidden_channels, hidden_channels * 2),
            Downsampling(hidden_channels * 2, hidden_channels * 4),
            Downsampling(hidden_channels * 4, hidden_channels * 8, stride=1),
            torch.nn.Conv2d(hidden_channels * 8, 1, kernel_size=4, padding=1)
        )
​
    def forward(self, x):
        return self.block(x)
​
​
class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            self.capacity = 0
            self.buffer = []
​
    def __call__(self, images):
        if self.buffer_size <= 0:  # non-used buffer
            return images
        returned_images = []
        for image in images:
            image = image.unsqueeze(dim=0)
            if self.capacity < self.buffer_size:
                self.capacity += 1
                self.buffer.append(image)
                returned_images.append(image)
            else:
                p = np.random.uniform()
                if p > 0.5:
                    index = np.random.randint(low=0, high=self.buffer_size)
                    drawn_image = self.buffer[index].clone()
                    self.buffer[index] = image
                    returned_images.append(drawn_image)
                else:
                    returned_images.append(image)
        return torch.cat(returned_images, dim=0)
​
​
class DoubleStableUNetGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config['hidden_channels']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.time_emb_dim = config['time_embedding']
​
        # Time embedding
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
            torch.nn.ReLU(True)
        )
​
        # Initial projection
        self.conv0 = torch.nn.Conv2d(self.in_channels, self.hidden_channels, 3, padding=1)
​
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
                                              # AttentionDownsamplingBlock(self.hidden_channels * 8,
                                              #                            self.hidden_channels * 8,
                                              #                            self.hidden_channels * 8,
                                              #                            self.time_emb_dim),
                                              # StableDownsamplingBlock(self.hidden_channels * 8,
                                              #                         self.hidden_channels * 8,
                                              #                         self.time_emb_dim),
                                              )
        self.up_block = torch.nn.Sequential(StableUpsamplingBlock(self.hidden_channels * 8,
                                                                  self.hidden_channels * 8,
                                                                  self.time_emb_dim),
                                            # AttentionUpsamplingBlock(self.hidden_channels * 16,
                                            #                          self.hidden_channels * 8,
                                            #                          self.hidden_channels * 8,
                                            #                          self.time_emb_dim),
                                            # StableUpsamplingBlock(self.hidden_channels * 16,
                                            #                       self.hidden_channels * 8,
                                            #                       self.time_emb_dim),
                                            # StableUpsamplingBlock(self.hidden_channels * 16,
                                            #                       self.hidden_channels * 8,
                                            #                       self.time_emb_dim),
                                            StableUpsamplingBlock(self.hidden_channels * 16,
                                                                  self.hidden_channels * 4,
                                                                  self.time_emb_dim),
                                            StableUpsamplingBlock(self.hidden_channels * 8,
                                                                  self.hidden_channels * 2,
                                                                  self.time_emb_dim),
                                            )
​
        self.feature_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(self.hidden_channels * 4, self.out_channels,
                                                                          kernel_size=4, stride=2, padding=1),
                                                 )
​
        self.conv1 = torch.nn.Conv2d(self.in_channels, self.hidden_channels, 3, padding=1)
​
        self.down_block1 = torch.nn.Sequential(StableDownsamplingBlock(self.hidden_channels,
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
                                               # AttentionDownsamplingBlock(self.hidden_channels * 8,
                                               #                            self.hidden_channels * 8,
                                               #                            self.hidden_channels * 8,
                                               #                            self.time_emb_dim),
                                               # StableDownsamplingBlock(self.hidden_channels * 8,
                                               #                         self.hidden_channels * 8,
                                               #                         self.time_emb_dim),
                                               )
        self.up_block1 = torch.nn.Sequential(StableUpsamplingBlock(self.hidden_channels * 8,
                                                                   self.hidden_channels * 8,
                                                                   self.time_emb_dim),
                                             # AttentionUpsamplingBlock(self.hidden_channels * 16,
                                             #                          self.hidden_channels * 8,
                                             #                          self.hidden_channels * 8,
                                             #                          self.time_emb_dim),
                                             # StableUpsamplingBlock(self.hidden_channels * 16,
                                             #                       self.hidden_channels * 8,
                                             #                       self.time_emb_dim),
                                             # StableUpsamplingBlock(self.hidden_channels * 16,
                                             #                       self.hidden_channels * 8,
                                             #                       self.time_emb_dim),
                                             StableUpsamplingBlock(self.hidden_channels * 16,
                                                                   self.hidden_channels * 4,
                                                                   self.time_emb_dim),
                                             StableUpsamplingBlock(self.hidden_channels * 8,
                                                                   self.hidden_channels * 2,
                                                                   self.time_emb_dim),
                                             )
​
        self.feature_block1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(self.hidden_channels * 4, self.out_channels,
                                                                           kernel_size=4, stride=2, padding=1),
                                                  )
​
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
        x = self.feature_block(x)
        output1 = Tensor(x)
​
        x = self.conv1(x)
        skips = []
        xPrev = Tensor(x)
        for down in self.down_block1:
            if isinstance(down, AttentionDownsamplingBlock):
                x = down(x, xPrev, t)
                xPrev = Tensor(x)
            else:
                xPrev = Tensor(x)
                x = down(x, t)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_block1, skips):
            if isinstance(up, AttentionUpsamplingBlock):
                x = up(x, xPrev, t)
                xPrev = Tensor(x)
            else:
                xPrev = Tensor(x)
                x = up(x, t)
            x = torch.cat([x, skip], dim=1)
        output2 = self.feature_block1(x)
​
        return output2, output1
​
​
class StableDiffuserGenerator(L.LightningModule):
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
        self.model = DoubleStableUNetGenerator(config)
        self.noise_scheduler = NoiseScheduler(config['time_steps'])
        self.device1 = config['device']
​
    def forward(self, x):
        for time in range(self.noise_scheduler.time_steps - 1, -1, -1):
            t = torch.Tensor([time]).type(torch.int64).to(self.device1)
            betas_t = get_index_from_list(self.noise_scheduler.scheduler, t, x.shape)
            sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
                self.noise_scheduler.sqrt_one_minus_alphas_cum_prod, t, x.shape)
            sqrt_recip_alphas_t = get_index_from_list(self.noise_scheduler.sqrt_recip_alphas, t, x.shape)
​
            # Call model (current image - noise prediction)
            model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * self.model(x, t)[0] / sqrt_one_minus_alphas_cum_prod_t)
            if time == 0:
                x = model_mean
            else:
                noise = torch.randn_like(x)
                posterior_variance_t = get_index_from_list(self.noise_scheduler.posterior_variance, t, x.shape)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
        return torch.clamp(x, -1, 1)
​
​
class DiffusionCycleGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config['lr']
        self.betas = config['betas']
        self.lambda_idt = config['lambda_idt']
        self.lambda_cycle = config['lambda_cycle']
        self.num_epochs = config['num_epochs']
        self.decay_epochs = config['decay_epochs']
        self.optimizer = config['optimizer']
        self.generator = config['generator_sd']
        self.discriminator = config['discriminator']
        self.automatic_optimization = False
        self.gen_PM = StableDiffuserGenerator(config)
        self.gen_MP = StableDiffuserGenerator(config)
        # self.disc_M = Discriminator(config)
        # self.disc_P = Discriminator(config)
        buffer_size = config['buffer_size']
        # self.buffer_fake_M = ImageBuffer(buffer_size)
        # self.buffer_fake_P = ImageBuffer(buffer_size)
        self.mean_init = config['mean_weight_init']
        self.std_init = config['std_weight_init']
        self.adversarial_criterion = torch.nn.functional.mse_loss
        self.recon_criterion = torch.nn.functional.l1_loss
        self.k_decay = config['k_decay']
        self.k = config['k_initial']
        self.k_minimum = config['k_minimum']
        self.starting_k_decay = config['starting_k_decay']
​
    def forward(self, x):
        t_images = self.gen_PM(x)
        return v2.Resize((256, 256), antialias=True)(t_images)
​
    def setup(self, stage):
        def init_fn(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d)):
                torch.nn.init.normal_(m.weight, self.mean_init, self.std_init)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
​
        if stage == "fit":
            for net in [self.gen_PM, self.gen_MP]:
                net.apply(init_fn)
            print("Model initialized.")
​
    def get_lr_scheduler(self, optimizer):
        def lr_lambda(epoch):
            len_decay_phase = self.num_epochs - self.decay_epochs + 1.0
            current_decay_step = max(0, epoch - self.decay_epochs + 1.0)
            val = 1.0 - current_decay_step / len_decay_phase
            return max(0.0, val)
​
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
​
    def configure_optimizers(self):
        opt_config = {
            "lr": self.lr,
            "betas": self.betas,
        }
        opt_gen = self.optimizer(list(self.gen_PM.parameters()) + list(self.gen_MP.parameters()), **opt_config)
        # opt_disc = self.optimizer(list(self.disc_M.parameters()) + list(self.disc_P.parameters()), **opt_config)
        optimizers = [opt_gen]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers
​
    def compute_k_adversarial_loss(self, input, discriminate):
        predicted_labels = discriminate(input)
        real_labels = torch.ones_like(predicted_labels[0])
        predicted_labels = sorted(predicted_labels,
                                  key=lambda sample: self.adversarial_criterion(sample, real_labels))
        predicted_labels = predicted_labels[:self.k]
        predicted_labels = torch.cat(predicted_labels, dim=0)
        real_labels = torch.ones_like(predicted_labels)
        return self.adversarial_criterion(predicted_labels, real_labels)
​
    def compute_adversarial_loss(self, input, discriminate, output):
        predicted_labels = discriminate(input)
        if output == 0:
            return self.adversarial_criterion(predicted_labels, torch.zeros_like(predicted_labels))
        else:
            return self.adversarial_criterion(predicted_labels, torch.ones_like(predicted_labels))
​
    def compute_discriminative_loss(self, real, fake, discriminate):
        return (0.5 * self.compute_adversarial_loss(real, discriminate, 1) +
                0.5 * self.compute_adversarial_loss(fake, discriminate, 0))
​
    def compute_losses(self, real_M, real_P,
                       # fake_M, fake_P,
                       noise1, noise2, noisy_images1, noisy_images2,
                       noise_pred_PM_M, noise_pred_PM_P,
                       noise_pred_MP_P, noise_pred_MP_M):
        gen_loss = (  # self.compute_k_adversarial_loss(noise_pred_MP_M, self.disc_M) +
            #             self.compute_k_adversarial_loss(noise_pred_PM_P, self.disc_P) +
                    self.lambda_cycle[0] * self.adversarial_criterion(noise1, noise_pred_PM_M) +
                    self.lambda_cycle[1] * self.adversarial_criterion(noise2, noise_pred_MP_P) +
                    self.lambda_cycle[2] * self.recon_criterion(noisy_images1, noise_pred_MP_M) +
                    self.lambda_cycle[3] * self.recon_criterion(noisy_images2, noise_pred_PM_P)
        )
        # disc_loss_M = self.compute_discriminative_loss(real_M, (self.buffer_fake_M(noise_pred_MP_M)).detach(),
        #                                                self.disc_M)
        # disc_loss_P = self.compute_discriminative_loss(real_P, (self.buffer_fake_P(noise_pred_PM_P)).detach(),
        #                                                self.disc_P)
        return gen_loss
​
    def training_step(self, batch, batch_idx):
        real_M = batch["monet"]
        real_P = batch["photo"]
        opt_gen = self.optimizers()
​
        bs = real_M.shape[0]
        time_steps = torch.randint(
            0, self.gen_PM.noise_scheduler.time_steps - 1, (bs,), device=real_M.device
        ).long()
        noise = torch.randn_like(real_M)
​
        noisy_images1 = self.gen_PM.noise_scheduler.forward_diffusion_sample(real_M, time_steps, noise=False, sampled_noise=noise)
        noise_pred_PM_M, noise_pred_PM_P = self.gen_PM.model(noisy_images1, time_steps)
​
        noisy_images2 = self.gen_MP.noise_scheduler.forward_diffusion_sample(real_P, time_steps, noise=False, sampled_noise=noise)
        noise_pred_MP_P, noise_pred_MP_M = self.gen_MP.model(noisy_images2, time_steps)
​
        gen_loss = self.compute_losses(real_M, real_P,
                                       noise, noise, noisy_images1,
                                       noisy_images2,
                                       noise_pred_PM_M, noise_pred_PM_P,
                                       noise_pred_MP_P, noise_pred_MP_M)
​
        self.toggle_optimizer(opt_gen)
        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)
​
        # self.toggle_optimizer(opt_disc)
        # opt_disc.zero_grad()
        # self.manual_backward(disc_loss_M)
        # self.manual_backward(disc_loss_P)
        # opt_disc.step()
        # self.untoggle_optimizer(opt_disc)
​
        metrics = {
            "gen_loss": gen_loss,
            # "disc_loss_M": disc_loss_M,
            # "disc_loss_P": disc_loss_P
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
​
    def validation_step(self, batch, batch_index):
        self.display_results(batch)
​
    def test_step(self, batch, batch_index):
        self.display_results(batch)
​
    def predict_step(self, batch, batch_index):
        return self(torch.randn_like(batch))
​
    def on_train_epoch_end(self):
        self.lr_schedulers().step()
        if self.current_epoch >= self.starting_k_decay:
            self.k = max(math.ceil(self.k * self.k_decay), self.k_minimum)
​
    def on_predict_epoch_end(self):
        predictions = self.trainer.predict_loop.predictions
        num_batches = len(predictions)
        batch_size = predictions[0].shape[0]
        last_batch_diff = batch_size - predictions[-1].shape[0]
        print(f"Number of images generated: {num_batches * batch_size - last_batch_diff}")
​
    def display_results(self, batch):
        real_P = torch.randn_like(batch)
        fake_M = self(real_P)
​
        title = f"Sample {self.current_epoch + 1}: Photo-to-Monet Translation"
​
        plot_images(torch.cat([fake_M], dim=0), lines=len(real_P), title=title)
