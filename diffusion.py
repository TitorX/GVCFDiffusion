from typing import Tuple, List, Union
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
import torch
import torch.nn as nn
import torchvision.transforms as T
from GVCFDiffusionMixin import GVCFDiffusionMixin, make_grid, randn_slerp, seed_everything
from tqdm import tqdm


class StableDiffusion(nn.Module, GVCFDiffusionMixin):
    """
    Stable diffusion model for text-to-image generation.
    """
    def __init__(self, device='cuda', sd_version='2.0', scheduler='ddim', compile=True):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        
        self.model_key = model_key

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        if compile:
            # compile for speed up
            self.unet = torch.compile(self.unet, mode="max-autotune", fullgraph=True)

        if scheduler == 'ddim':
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
            self.num_inference_steps = 50
        elif scheduler == 'ddpm':
            self.scheduler = DDPMScheduler.from_pretrained(model_key, subfolder='scheduler', steps_offset=0)
            self.num_inference_steps = 1000
        else:
            raise ValueError(f'Scheduler {scheduler} not supported')

        self.guidance_scale = 7.5

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def text2img(self, prompts, negative_prompts='', batch_size=32):
        """
        Generate (512, 512) images from text prompts. Using original stable-diffusion model.
        """

        if isinstance(prompts, str):
            prompts = [prompts] * batch_size

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * batch_size

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Define panorama grid and get views
        latent = torch.randn((batch_size, self.unet.config.in_channels, 64, 64), device=self.device)

        self.scheduler.set_timesteps(self.num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latent] * 2)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = self.scheduler.step(noise_pred, t, latent)['prev_sample']

        # Img latents -> imgs
        imgs = self.decode_latents(latent)  # [B, 3, 512, 512]
        imgs = [T.ToPILImage()(img.cpu()) for img in imgs]
        return imgs

    def _batch_forward(self, latent, text_embeds, t, batch_size):
        epsilon_pred = torch.zeros_like(latent)

        # split into batch size
        latent = latent.split(batch_size)

        # predict the noise residual
        for i, l_in in enumerate(latent):
            # expand the latents and text embeds if we are doing classifier-free guidance
            # to avoid doing two forward passes.
            t_in = torch.repeat_interleave(
                text_embeds, len(l_in), dim=0
            )
            l_in =  torch.cat([l_in] * 2)

            e_pred = self.unet(l_in, t, encoder_hidden_states=t_in)['sample']

            # perform guidance
            e_pred_uncond, e_pred_cond = e_pred.chunk(2)
            e_pred = e_pred_uncond + self.guidance_scale * (e_pred_cond - e_pred_uncond)

            epsilon_pred[i*batch_size:(i+1)*batch_size] = e_pred

        return epsilon_pred

    @torch.no_grad()
    def text2panorama(
        self,
        image_size: Tuple[int, int],
        prompts: Union[str, List[str]],
        stride: Tuple[int, int],
        step_func: str,
        slerp: float = 0.,
        negative_prompts: Union[str, List[str]] = '',
        batch_size: int = 64
    ):
        """
        Sample panorama images from text prompts.

        Args:
            image_size (Tuple[int, int]): Image size (h, w).
            prompts (Union[str, List[str]]): Text prompts.
            step_func (str): Step function to use. [gf, vcf, gvcf].
            negative_prompts (Union[str, List[str]]): Negative text prompts. Defaults to ''.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, "Image size should be divisible by 8."
        assert stride[0] % 8 == 0 and stride[1] % 8 == 0, "Stride should be divisible by 8."

        latent_size = image_size[0] // 8, image_size[1] // 8
        stride = (stride[0] // 8, stride[1] // 8)
        if isinstance(self.unet.config.sample_size, int):
            sample_size = (self.unet.config.sample_size,) * 2
        else:
            sample_size = self.unet.config.sample_size

        if slerp > 0:
            latent = randn_slerp((4, *latent_size), sample_size, slerp).to(self.device)
        else:
            latent = torch.randn((4, *latent_size), device=self.device)

        # set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, leave=False)):
                latent_patches = make_grid(latent, sample_size, stride)

                epsilon_pred = self._batch_forward(
                    latent_patches,
                    text_embeds,
                    t,
                    batch_size
                )

                if step_func == 'gf':
                    latent = self._gf_step(epsilon_pred, latent_patches, t, latent, stride)
                elif step_func == 'vcf':
                    latent = self._vcf_step(epsilon_pred, latent_patches, t, latent, stride)
                elif step_func == 'gvcf':
                    latent = self._gvcf_step(epsilon_pred, latent_patches, t, latent, stride)
                else:
                    raise ValueError(f'Step function {step_func} not supported.')

        latent.unsqueeze_(0)
        # Img latents -> imgs
        imgs = self.decode_latents(latent)
        img = T.ToPILImage()(imgs[0].cpu())
        return img
