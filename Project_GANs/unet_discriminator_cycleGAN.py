import math
import sys

import numpy as np
import pytorch_lightning as L
import torch
from torch import Tensor

from configs import DATALOADER_CONFIG
from cutmix_mixup import CutMix
from utils import plot_images


class Downsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1, norm=True, lrelu=True):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels,
                                                         kernel_size=kernel_size, stride=stride,
                                                         padding=padding, bias=not norm))
        if norm:
            self.block.append(torch.nn.InstanceNorm2d(out_channels, affine=True))
        if lrelu is True:
            self.block.append(torch.nn.LeakyReLU(0.2, True) if lrelu else torch.nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Upsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1, output_padding=0, dropout=False):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, out_channels, bias=False,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding, output_padding=output_padding),
                                         torch.nn.InstanceNorm2d(out_channels, affine=True))
        if dropout:
            self.block.append(torch.nn.Dropout(0.5))
        self.block.append(torch.nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class UNetGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_channels = config['hidden_channels']
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        self.downsampling_path = torch.nn.Sequential(Downsampling(in_channels, hidden_channels, norm=False,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels, hidden_channels * 2,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels * 2, hidden_channels * 4,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels * 4, hidden_channels * 8,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels * 8, hidden_channels * 8,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels * 8, hidden_channels * 8,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels * 8, hidden_channels * 8,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     Downsampling(hidden_channels * 8, hidden_channels * 8, norm=False,
                                                                  kernel_size=kernel_size, stride=stride,
                                                                  padding=padding),
                                                     )
        self.upsampling_path = torch.nn.Sequential(Upsampling(hidden_channels * 8, hidden_channels * 8, dropout=True,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   Upsampling(hidden_channels * 16, hidden_channels * 8, dropout=True,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   Upsampling(hidden_channels * 16, hidden_channels * 8, dropout=True,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   Upsampling(hidden_channels * 16, hidden_channels * 8,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   Upsampling(hidden_channels * 16, hidden_channels * 4,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   Upsampling(hidden_channels * 8, hidden_channels * 2,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   Upsampling(hidden_channels * 4, hidden_channels,
                                                              kernel_size=kernel_size, stride=stride,
                                                              padding=padding),
                                                   )
        self.feature_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(hidden_channels * 2, out_channels,
                                                                          kernel_size=kernel_size, stride=stride,
                                                                          padding=padding),
                                                 torch.nn.Tanh())

    def forward(self, x):
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.upsampling_path, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x)


class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_channels = config['hidden_channels']
        in_channels = config['in_channels']
        kernel_size = config['kernel_size']
        stride = config['disc_stride']
        padding = config['padding']
        self.downsampling_path = torch.nn.Sequential(
            Downsampling(in_channels, hidden_channels, norm=False,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding),
            Downsampling(hidden_channels, hidden_channels * 2,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding),
            Downsampling(hidden_channels * 2, hidden_channels * 4,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding),
            Downsampling(hidden_channels * 4, hidden_channels * 8,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding),
            Downsampling(hidden_channels * 8, hidden_channels * 8,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding),
        )
        self.upsampling_path = torch.nn.Sequential(
            Upsampling(hidden_channels * 8, hidden_channels * 8,
                       kernel_size=kernel_size, stride=stride,
                       padding=padding),
            Upsampling(hidden_channels * 16, hidden_channels * 4,
                       kernel_size=kernel_size, stride=stride,
                       padding=padding),
            Upsampling(hidden_channels * 8, hidden_channels * 2,
                       kernel_size=kernel_size, stride=stride,
                       padding=padding),
            Upsampling(hidden_channels * 4, hidden_channels,
                       kernel_size=kernel_size, stride=stride,
                       padding=padding),
        )

        self.feature_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(hidden_channels * 2, 1,
                                                                          kernel_size=kernel_size, stride=stride,
                                                                          padding=padding),

                                                 torch.nn.Tanh())
        self.extract_layer = torch.nn.Conv2d(hidden_channels * 8, hidden_channels * 4,
                                             kernel_size=kernel_size, stride=1,
                                             padding=padding)

    def forward(self, x):
        skips = []
        for down in self.downsampling_path:
            x = down(x)
            skips.append(x)
        last = Tensor(skips[-1])
        skips = reversed(skips[:-1])
        for up, skip in zip(self.upsampling_path, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        return self.feature_block(x), self.extract_layer(last)


class ImageBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            self.capacity = 0
            self.buffer = []

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


class CycleGAN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config['lr']
        self.betas = config['betas']
        self.lambda_idt = config['lambda_idt']
        self.lambda_cycle = config['lambda_cycle']
        self.num_epochs = config['num_epochs']
        self.decay_epochs = config['decay_epochs']
        self.optimizer = config['optimizer']
        self.generator = config['generator']
        self.discriminator = config['discriminator']
        self.automatic_optimization = False
        generator = getattr(sys.modules[__name__], self.generator)
        discriminator = getattr(sys.modules[__name__], self.discriminator)
        self.gen_PM = generator(config)
        self.gen_MP = generator(config)
        self.disc_M = discriminator(config)
        self.disc_P = discriminator(config)
        buffer_size = config['buffer_size']
        self.buffer_fake_M = ImageBuffer(buffer_size)
        self.buffer_fake_P = ImageBuffer(buffer_size)
        self.mean_init = config['mean_weight_init']
        self.std_init = config['std_weight_init']
        self.adversarial_criterion = torch.nn.functional.mse_loss
        self.recon_criterion = torch.nn.functional.l1_loss
        self.k_decay = config['k_decay']
        self.k = config['k_initial']
        self.k_minimum = config['k_minimum']
        self.starting_k_decay = config['starting_k_decay']
        self.div_idt = config['div_idt']

    def forward(self, x):
        return self.gen_PM(x)

    def setup(self, stage):
        def init_fn(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d)):
                torch.nn.init.normal_(m.weight, self.mean_init, self.std_init)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        if stage == "fit":
            for net in [self.gen_PM, self.disc_M]:
                net.apply(init_fn)
            print("Model initialized.")

    def get_lr_scheduler(self, optimizer):
        def lr_lambda(epoch):
            len_decay_phase = self.num_epochs - self.decay_epochs + 1.0
            current_decay_step = max(0, epoch - self.decay_epochs + 1.0)
            val = 1.0 - current_decay_step / len_decay_phase
            return max(0.0, val)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        opt_config = {
            "lr": self.lr,
            "betas": self.betas,
        }
        opt_gen = self.optimizer(list(self.gen_PM.parameters()) + list(self.gen_MP.parameters()), **opt_config)
        opt_disc = self.optimizer(list(self.disc_M.parameters()) + list(self.disc_P.parameters()), **opt_config)
        optimizers = [opt_gen, opt_disc]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
        return optimizers, schedulers

    def compute_k_adversarial_loss(self, input, discriminate):
        predicted_labels = discriminate(input)[0]
        real_labels = torch.ones_like(predicted_labels[0])
        predicted_labels = sorted(predicted_labels,
                                  key=lambda sample: self.adversarial_criterion(sample, real_labels))
        predicted_labels = predicted_labels[:self.k]
        predicted_labels = torch.cat(predicted_labels, dim=0)
        real_labels = torch.ones_like(predicted_labels)
        return self.adversarial_criterion(predicted_labels, real_labels)

    def compute_adversarial_loss(self, input, discriminate, cutmix_labels):
        predicted_labels = discriminate(input)
        real_labels = torch.zeros_like(predicted_labels[1])
        cutmix_labels = torch.cat([cutmix_labels.unsqueeze(0).unsqueeze(0)]
                                  * predicted_labels[0].shape[0], dim=0)
        return (self.adversarial_criterion(predicted_labels[0], cutmix_labels) +
                self.div_idt * self.recon_criterion(predicted_labels[1], real_labels))

    def compute_idt_loss(self, real, idt):
        idt_loss = self.recon_criterion(idt, real)
        return self.lambda_cycle[0] * idt_loss

    def compute_cycle_loss(self, real, recon):
        cycle_loss = self.recon_criterion(recon, real)
        return self.lambda_cycle[1] * cycle_loss

    def compute_discriminative_loss(self, real, fake, discriminate):
        cutmix_labels, c1, c2, c3, c4 = CutMix(DATALOADER_CONFIG['image_size'])
        real_1 = real.clone()
        real_1[:, :, c1:c2, c3:c4] = fake[:, :, c1:c2, c3:c4].clone()
        disc_real = discriminate(real)
        disc_fake = discriminate(fake)
        return (0.5 * self.compute_adversarial_loss(real_1, discriminate, cutmix_labels.to(real.device)) +
                0.5 * self.adversarial_criterion(disc_real[0], torch.ones_like(disc_real[0])) +
                0.5 * self.adversarial_criterion(disc_fake[0], torch.zeros_like(disc_fake[0])) +
                0.5 * self.recon_criterion(disc_real[1], torch.ones_like(disc_real[1])) +
                0.5 * self.recon_criterion(disc_fake[1], torch.zeros_like(disc_fake[1]))
                ,
                self.recon_criterion(fake * cutmix_labels.to(real.device)
                                     + real * (1 - cutmix_labels.to(real.device)), real_1),
                )

    def compute_losses(self, real_M, real_P, fake_P, fake_M, idt_M, idt_P, recon_M, recon_P):
        gen_loss = (self.compute_k_adversarial_loss(fake_M, self.disc_M)
                    + self.compute_k_adversarial_loss(fake_P, self.disc_P)
                    + self.lambda_idt * self.compute_idt_loss(real_M, idt_M)
                    + self.lambda_idt * self.compute_idt_loss(real_P, idt_P)
                    + self.compute_cycle_loss(real_M, recon_M)
                    + self.compute_cycle_loss(real_P, recon_P)
                    )
        disc_loss_M = self.compute_discriminative_loss(real_M, (self.buffer_fake_M(fake_M)).detach(), self.disc_M)
        disc_loss_P = self.compute_discriminative_loss(real_P, (self.buffer_fake_P(fake_P)).detach(), self.disc_P)
        return gen_loss, disc_loss_M[1] + disc_loss_P[1], disc_loss_M[0], disc_loss_P[0]

    def training_step(self, batch, batch_idx):
        real_M = batch["monet"]
        real_P = batch["photo"]
        opt_gen, opt_disc = self.optimizers()

        fake_M = self.gen_PM(real_P)
        fake_P = self.gen_MP(real_M)

        idt_M = self.gen_PM(real_M)
        idt_P = self.gen_MP(real_P)

        recon_M = self.gen_PM(fake_P)
        recon_P = self.gen_MP(fake_M)

        gen_loss, gen_loss_mixed, disc_loss_M, disc_loss_P = self.compute_losses(real_M, real_P, fake_P, fake_M,
                                                                                 idt_M, idt_P,
                                                                                 recon_M, recon_P,
                                                                                 )

        self.toggle_optimizer(opt_gen)
        opt_gen.zero_grad()
        self.manual_backward(gen_loss + gen_loss_mixed)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)

        self.toggle_optimizer(opt_disc)
        opt_disc.zero_grad()
        self.manual_backward(disc_loss_M)
        self.manual_backward(disc_loss_P)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        metrics = {
            "gen_loss": gen_loss,
            "disc_loss_M": disc_loss_M,
            "disc_loss_P": disc_loss_P
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_index):
        self.display_results(batch)

    def test_step(self, batch, batch_index):
        self.display_results(batch)

    def predict_step(self, batch, batch_index):
        return self(batch)

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()
        if self.current_epoch >= self.starting_k_decay:
            self.k = max(math.ceil(self.k * self.k_decay), self.k_minimum)

    def on_predict_epoch_end(self):
        predictions = self.trainer.predict_loop.predictions
        num_batches = len(predictions)
        batch_size = predictions[0].shape[0]
        last_batch_diff = batch_size - predictions[-1].shape[0]
        print(f"Number of images generated: {num_batches * batch_size - last_batch_diff}")

    # def on_validation_end(self):
    #     logged_values = self.trainer.progress_bar_metrics
    #     print(
    #         f"\nEpoch {self.current_epoch + 1}",
    #         *[f"{key}: {val:.5f}" for key, val in logged_values.items()],
    #         sep=" - ",
    #     )

    def display_results(self, batch):
        real_P = batch
        fake_M = self(real_P)

        title = f"Sample {self.current_epoch + 1}: Photo-to-Monet Translation"

        plot_images(torch.cat([real_P, fake_M], dim=0), lines=len(real_P), title=title)
