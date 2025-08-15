# -*- coding: utf-8 -*-
import argparse
import json
import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFile
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
        x = x.permute(0, 2, 3, 1).view(-1, c)
        y = self.network(x)
        # y: (b h w) c -> b h w c -> b c h w
        y = y.view(b, h, w, -1).permute(0, 3, 1, 2)
        return y


# LCTN Sampler
class LCTNSampler(object):
    
    def __init__(self, config):
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.force_cpu)
        pipeline = StableDiffusionPipeline.from_pretrained(config.sd_path)
        self.scheduler = pipeline.scheduler
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder.eval()
        self.unet = pipeline.unet.eval()
        self.vae = pipeline.vae.eval()
        self.lctn = LCTN(9320, 4)
        self.lctn.load_state_dict(torch.load(config.lctn_path))
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.lctn.requires_grad_(True)
        del pipeline
        self.scheduler, \
        self.tokenizer, \
        self.text_encoder, \
        self.unet, \
        self.vae, \
        self.lctn = self.accelerator.prepare(
            self.scheduler,
            self.tokenizer,
            self.text_encoder,
            self.unet,
            self.vae,
            self.lctn
        )
        self.latent_feature_blocks = self._register_unet_hooks()
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
    
    def get_image_transform(self, image_size=768):
        transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        return transform
    
    def get_latent_features(self, size=96, interpolation='bilinear', do_cfg=True):
        latent_features = []
        for block in self.latent_feature_blocks:
            block_output = block.output.chunk(2)[1] if do_cfg else block.output
            interpolated_features = nn.functional.interpolate(block_output, size=size, mode=interpolation)
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
    
    def get_visuals(self, images, thumbnail_size):
        for i in range(len(images)):
            image = images[i].detach().cpu().float()
            image = nn.functional.interpolate(image, size=thumbnail_size, mode='bilinear')
            image = (image / 2 + 0.5).clamp(0, 1)
            image = make_grid(image, nrow=1)
            images[i] = image[:3, :, :]
        grid_image = torch.cat(images, dim=2)
        grid_image = T.ToPILImage()(grid_image)
        return grid_image
    
    def pseudo_noising(self, latents):
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
    
    def sample(
        self,
        prompt,
        sketch,
        image_size=768,
        guidance_scale=8.0,
        noising_scale=0.8,
        steps=250
    ):
        assert isinstance(prompt, str)
        assert isinstance(sketch, str) or isinstance(sketch, Image.Image)
        assert image_size % 8 == 0
        assert noising_scale >= 0.0 and noising_scale <= 1.0
        assert steps > 0
        
        image_transform = self.get_image_transform(image_size)
        do_cfg = guidance_scale > 1.0
        
        # 1. encode input prompt and sketch
        if isinstance(sketch, str):
            sketch = image_transform(Image.open(sketch).convert('RGB')).unsqueeze(0).to(self.accelerator.device)
        else:
            sketch = image_transform(sketch.convert('RGB')).unsqueeze(0).to(self.accelerator.device)
        with torch.no_grad():
            c = self.encode_prompt(prompt, do_cfg)
            e = self.vae.encode(sketch).latent_dist.sample() * self.vae.config.scaling_factor
        
        # 2. translate latent code from sketch to image
        noisy_e, t, timesteps = self.pseudo_noising(e)
        noisy_e = torch.cat([noisy_e] * 2)
        
        with torch.no_grad():
            noise_pred = self.unet(noisy_e, timesteps, c).sample
        
        F = self.get_latent_features(size=noisy_e.shape[2], do_cfg=do_cfg)
        p = self.get_positional_encoding(t)
        
        with torch.no_grad():
            x_pred = self.lctn(F, t, p)
        
        # 3. prepare timesteps
        self.scheduler.set_timesteps(steps, device=self.accelerator.device)
        timesteps = self.scheduler.timesteps
        
        # 4. prepare latents
        noise = torch.randn_like(x_pred).to(self.accelerator.device)
        
        index = min(int(round((1 - noising_scale) * steps)), steps - 1)
        latent = self.scheduler.add_noise(x_pred, noise, timesteps[index:index+1])
        # latent = (latent - latent.mean()) / latent.std()
        latent0 = latent
        
        # 5. denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2) if do_cfg else latent
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=c).sample
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            print(f'\rRunning inference step... {i + 1}/{steps}', end='')
        print('')
        
        # ---------------- free memory ------------
        self.text_encoder = self.text_encoder.cpu()
        self.unet = self.unet.cpu()
        self.lctn = self.lctn.cpu()
        del self.text_encoder
        del self.unet
        del self.lctn
        torch.cuda.empty_cache()
        # -----------------------------------------
        
        # 6. post-processing
        z0 = 1 / self.vae.config.scaling_factor * latent
        with torch.no_grad():
            image = self.vae.decode(z0).sample
        
        # 7. prepare visuals
        image_pil = self.get_visuals([image], image.shape[2])
        merged_visuals_pil = self.get_visuals([sketch, e, x_pred, latent0, z0, image], z0.shape[2])
        
        return {'image': image_pil, 'merged_visuals': merged_visuals_pil}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt')
    parser.add_argument('--sketch', type=str, default=None, help='Path of sketch')
    parser.add_argument('--image_size', type=int, default=768, help='Image size')
    parser.add_argument('--guidance_scale', type=float, default=8.0, help='Classifier-free guidance scale')
    parser.add_argument('--noising_scale', type=float, default=0.8, help='Latent noising scale')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--sd_path', type=str, default='stabilityai/stable-diffusion-2-1', help='Name or path of pretrained LDM')
    parser.add_argument('--lctn_path', type=str, default='./lctn.pth', help='Path of pretrained LCTN')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16', 'fp8'], help='Use mixed precision sampling')
    parser.add_argument('--force_cpu', action='store_true', help='Force execution on CPU')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    config = parser.parse_args()
    return config


# main
if __name__ == '__main__':
    config = args()
    if config.seed is not None:
        torch.manual_seed(config.seed)
    lctns = LCTNSampler(config)
    output = lctns.sample(
        config.prompt,
        config.sketch,
        config.image_size,
        config.guidance_scale,
        config.noising_scale,
        config.steps
    )
    fname = os.path.splitext(os.path.basename(config.sketch))[0]
    config_path = os.path.join(config.output_dir, fname + '_config.json')
    image_path = os.path.join(config.output_dir, fname + '.jpg')
    visuals_path = os.path.join(config.output_dir, fname + '_visuals.jpg')
    if not os.path.isdir(config.output_dir): os.makedirs(config.output_dir)
    with open(config_path, 'w') as fp: json.dump(vars(config), fp, indent=2)
    output['image'].save(image_path)
    output['merged_visuals'].save(visuals_path)
