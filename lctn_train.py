# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, optimization
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid


ImageFile.LOAD_TRUNCATED_IMAGES = True


# Latent Code Translation Network (LCTN)
class LCTN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LCTN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        )
        self._init_params()
    
    def _init_params(self, gain=0.02):
        for _, module in self.network.named_modules():
            classname = module.__class__.__name__
            if hasattr(module, 'weight'):
                if classname.find('Linear') != -1 or classname.find('Conv') != -1:
                    nn.init.kaiming_uniform_(module.weight)
                elif classname.find('BatchNorm') != -1:
                    nn.init.normal_(module.weight, 1.0, gain)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, F, t, p):
        # F (b c h w): concatenated latent feature maps F(z_t|t,c) of the denoising U-Net
        # t (b c h w): noise level
        # p (b c h w): positional encoding
        x = torch.cat((F, t, p), dim=1)
        b, c, h, w = x.size()
        # x: b c h w -> b h w c -> (b h w) c
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.network(x)
        # y: (b h w) c -> b h w c -> b c h w
        y = y.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return y


# LCTN Dataset
class LCTNDataset(Dataset):
    
    def __init__(self, data_root, target_size=768, interpolation=T.InterpolationMode.BILINEAR):
        super(LCTNDataset, self).__init__()
        self.data_root = data_root
        self.data = pd.read_csv(os.path.join(data_root, 'data.csv'))
        self.transforms = T.Compose([
            T.Resize(target_size, interpolation=interpolation),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, sketch_path, prompt = self.data.iloc[index]
        image = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')
        sketch = Image.open(os.path.join(self.data_root, sketch_path)).convert('RGB')
        return {
            'image': self.transforms(image),
            'sketch': self.transforms(sketch),
            'prompt': prompt
        }


# LCTN Trainer
class LCTNTrainer(object):
    
    def __init__(self, config):
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.force_cpu)
        pipeline = StableDiffusionPipeline.from_pretrained(config.sd_path)
        self.scheduler = pipeline.scheduler
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder.eval()
        self.unet = pipeline.unet.eval()
        self.vae = pipeline.vae.eval()
        self.lctn = LCTN(9320, 4)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.lctn.requires_grad_(True)
        del pipeline
        self.dataloader = DataLoader(
            LCTNDataset(config.data_root, config.image_size),
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers
        )
        self.optimizer = torch.optim.Adam(self.lctn.parameters(), lr=config.lr)
        self.lr_scheduler = optimization.get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)
        self.scheduler, \
        self.tokenizer, \
        self.text_encoder, \
        self.unet, \
        self.vae, \
        self.lctn, \
        self.dataloader, \
        self.optimizer, \
        self.lr_scheduler = self.accelerator.prepare(
            self.scheduler,
            self.tokenizer,
            self.text_encoder,
            self.unet,
            self.vae,
            self.lctn,
            self.dataloader,
            self.optimizer,
            self.lr_scheduler
        )
        self.latent_feature_blocks = self._register_unet_hooks()
        self.best_iter = 0
        self.best_loss = 0
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.config = config
    
    def _register_unet_hooks(self):
        def hook(module, x, y):
            y = y[0] if isinstance(y, tuple) else y
            y = y.sample.float() if isinstance(y, dict) else y.float()
            setattr(module, 'output', y)
        block_ids = [0, 1, 2]
        latent_feature_blocks = []
        # [0, 1, 2] -> (ldm-down) 2, 4, 8
        for index, block in enumerate(self.unet.down_blocks):
            if index in block_ids:
                block.register_forward_hook(hook)
                latent_feature_blocks.append(block)
        # [0, 1, 2] -> (ldm-mid) 0, 1, 2
        for block in self.unet.mid_block.attentions + self.unet.mid_block.resnets:
            block.register_forward_hook(hook)
            latent_feature_blocks.append(block)
        # [0, 1, 2] -> (ldm-up) 2, 4, 8
        for index, block in enumerate(self.unet.up_blocks):
            if index in block_ids:
                block.register_forward_hook(hook)
                latent_feature_blocks.append(block)
        return latent_feature_blocks
    
    def encode_prompt(self, prompt, do_cfg=True):
        assert isinstance(prompt, str) or isinstance(prompt, list)
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        prompt_input_ids = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids
        prompt_embeds = self.text_encoder(prompt_input_ids.to(self.accelerator.device))[0]
        if do_cfg:
            negative_prompt_input_ids = self.tokenizer(
                [''] * batch_size,
                padding='max_length',
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors='pt'
            ).input_ids
            negative_prompt_embeds = self.text_encoder(negative_prompt_input_ids.to(self.accelerator.device))[0]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds
    
    def add_noise(self, latents):
        # latents (b c h w)
        assert isinstance(latents, torch.Tensor) and len(latents.shape) == 4
        n = 1
        b = latents.shape[0]
        noise = torch.randn_like(latents).to(self.accelerator.device)
        timesteps = torch.randint(0, n, (b,)).to(self.accelerator.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        sqrt_one_minus_alphas_cumprod = ((1 - self.scheduler.alphas_cumprod[timesteps.cpu()]) ** 0.5).flatten().to(self.accelerator.device)
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(noise.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
        noise_levels = sqrt_one_minus_alphas_cumprod * noise
        return noisy_latents, noise_levels, timesteps
    
    def get_latent_features(self, size=96, interpolation='bilinear'):
        latent_features = []
        for block in self.latent_feature_blocks:
            interpolated_features = nn.functional.interpolate(block.output, size=size, mode=interpolation)
            latent_features.append(interpolated_features)
            del block.output
        return torch.cat(latent_features, dim=1)
    
    def get_positional_encoding(self, t, n=None):
        # t (b c h w): noise level
        # n: number of encoding blocks
        n = n or len(self.latent_feature_blocks)
        positional_encoding = [torch.sin(2 * torch.pi * t * 2 ** -l) for l in range(n)]
        positional_encoding = torch.cat(positional_encoding, dim=1)
        return positional_encoding
    
    def train(self):
        dataloader_iterator = iter(self.dataloader)
        dataloader_index = 0
        for iteration in range(1, self.config.steps + 1):
            batch = next(dataloader_iterator)
            image = batch['image']
            sketch = batch['sketch']
            prompt = batch['prompt']
            with torch.no_grad():
                x = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
                e = self.vae.encode(sketch).latent_dist.sample() * self.vae.config.scaling_factor
                c = self.encode_prompt(prompt, do_cfg=False)
            noisy_e, t, timesteps = self.add_noise(e)
            with torch.no_grad():
                noise_pred = self.unet(noisy_e, timesteps, c).sample
            F = self.get_latent_features(size=e.shape[2])
            p = self.get_positional_encoding(t)
            x_pred = self.lctn(F, t, p)
            loss = torch.nn.functional.mse_loss(x, x_pred)
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.log(iteration, loss, [sketch, e, noisy_e, t, noise_pred, x_pred, x, image], x.shape[2])
            dataloader_index = (dataloader_index + 1) % len(self.dataloader)
            if dataloader_index == 0:
                dataloader_iterator = iter(self.dataloader)
    
    def log(self, iteration, loss, images, thumbnail_size):
        w_iter = len(str(self.config.steps))
        output_dir = os.path.join(self.config.output_root, self.timestamp)
        images_dir = os.path.join(output_dir, 'images')
        config_path = os.path.join(output_dir, 'config.json')
        last_state_path = os.path.join(output_dir, 'last_state.pt')
        best_state_path = os.path.join(output_dir, 'best_state.pt')
        best_checkpoint_path = os.path.join(output_dir, 'lctn.pth')
        for directory in [output_dir, images_dir]:
            if not os.path.isdir(directory):
                os.makedirs(directory)
        if not os.path.isfile(config_path):
            with open(config_path, 'w') as fp:
                json.dump(vars(self.config), fp, indent=2)
        if iteration == 1 or loss.item() < self.best_loss:
            self.best_iter = iteration
            self.best_loss = loss.item()
            torch.save({
                'lctn_state_dict': self.lctn.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'step': iteration,
                'loss': loss.item()
            }, best_state_path)
            torch.save(self.lctn.state_dict(), best_checkpoint_path)
            for i in range(len(images)):
                image = images[i].detach().cpu().float()
                image = nn.functional.interpolate(image, size=thumbnail_size, mode='bilinear')
                image = (image / 2 + 0.5).clamp(0, 1)
                image = make_grid(image, nrow=1)
                images[i] = image[:3, :, :]
            grid_image = torch.cat(images, dim=2)
            grid_image = T.ToPILImage()(grid_image)
            grid_image.save(os.path.join(images_dir, f'iter_{str(iteration).zfill(w_iter)}_loss_{loss.item():.4f}.jpg'))
        if iteration == 1 or iteration == self.config.steps or iteration % self.config.output_freq == 0:
            torch.save({
                'lctn_state_dict': self.lctn.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'step': iteration,
                'loss': loss.item()
            }, last_state_path)
            print(
                f'[TRAIN] ITER: {iteration:{w_iter}d}/{self.config.steps} | LOSS: {loss.item():.4f} |',
                f'BEST_ITER: {self.best_iter:{w_iter}d} | BEST_LOSS: {self.best_loss:.4f} |'
            )


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_path', type=str, default='stabilityai/stable-diffusion-2-1', help='Name or path of pretrained LDM')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16', 'fp8'], help='Use mixed precision training')
    parser.add_argument('--force_cpu', action='store_true', help='Force execution on CPU')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--image_size', type=int, default=768, help='Image size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--shuffle', action='store_true', help='Reshuffle data at every epoch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of subprocesses for data loading')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--steps', type=int, default=1, help='Number of training steps')
    parser.add_argument('--output_freq', type=int, default=1, help='Output frequency')
    parser.add_argument('--output_root', type=str, default='./output', help='Output directory')
    config = parser.parse_args()
    return config


# main
if __name__ == '__main__':
    config = args()
    lctnt = LCTNTrainer(config)
    lctnt.train()
