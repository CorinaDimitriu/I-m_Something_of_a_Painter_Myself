import torch

DATALOADER_CONFIG = {
    "path_paintings": ".\\gan-getting-started-old\\monet_jpg",
    "path_photos": ".\\gan-getting-started-old\\photo_jpg",
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": True,
    "sample_size": 5,
    "batch_size": 1,
    "val_batch_size": 1,
    "eval_batch_size": 7038,
    "len_dataset": 7038,
    "image_size": 256,
}

MODEL_CONFIG = {
    "device": torch.device("cuda"),
    "hidden_channels": 64,
    "in_channels": 3,
    "out_channels": 3,
    "optimizer": torch.optim.Adam,
    "lr": 2e-4,
    "betas": (0.5, 0.999),
    "lambda_idt": 0.5,
    "div_idt": 1.0,
    "lambda_cycle": (10, 10, 10, 10),  # (M-P-M, P-M-P)
    "buffer_size": 100,
    # total number of epochs
    "num_epochs": 18,
    # number of epochs before starting to decay the learning rate
    "decay_epochs": 18,
    "kernel_size": 4,
    "stride": 2,
    "disc_stride": 2,
    "padding": 1,
    "mean_weight_init": 0.0,
    "std_weight_init": 0.02,
    "generator": "UNetGenerator",
    "generator_sd": "StableDiffuser",
    "discriminator": "Discriminator",
    "k_decay": 0.99,
    "k_initial": DATALOADER_CONFIG['batch_size'],
    "k_minimum": int(DATALOADER_CONFIG['batch_size'] * 2 / 3),
    "starting_k_decay": 0,
    "seed": 0,
    "time_steps": 300,
    "time_embedding": 64,
    "mix_multiplier": 5
}

TRAIN_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    "precision": 32,
    "devices": 1,
    "enable_checkpointing": True,
    "logger": False,
    "max_epochs": MODEL_CONFIG["num_epochs"],
    "max_time": {"hours": 4, "minutes": 55},
    "limit_val_batches": 1,
    "limit_train_batches": 1.0,
    "limit_predict_batches": 1.0,
    "num_sanity_val_steps": 0,
    "check_val_every_n_epoch": 5,
}


class DiffusionUNetConfig:
    def __init__(self, **config):
        self.sample_size = config['image_size']
        self.hidden_channels = config['hidden_channels']
        self.in_channels = config['in_channels']
