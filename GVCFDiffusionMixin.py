from typing import Tuple, Union
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )


def randn_slerp(shape, patch_size, alpha):
    C, H, W = shape

    assert H % patch_size[0] == 0 and W % patch_size[1] == 0, "Image size should be divisible by 64."

    # number of patches
    NROW, NCOL = H // patch_size[0], W // patch_size[1]

    # random noise
    latent = torch.randn(shape)

    # style guidance noise, repeat for whole latent
    z = torch.randn((C, *patch_size)).repeat(1, NROW, NCOL)

    # return norm(latent, z, alpha)
    return slerp(latent, z, alpha)


def make_grid(
    image: torch.Tensor,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int]
) -> torch.Tensor:
    """
    Split an image into a grid of patches with stride.

    The image size / patch size / stride should be carefully chosen to be divisible.
    Which means: image size = n * stride + (patch size - stride)

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W).
        patch_size (Tuple[int, int]): Patch size.
        stride (Tuple[int, int]): Stride of the patch.

    Returns:
        torch.Tensor: patches tensor of shape (num_patches, C, patch_size[0], patch_size[1]).
    """
    C, H, W = image.shape

    # Check if the image size is divisible by stride and patch size
    assert (H - patch_size[0]) % stride[0] == 0, f"Height {H} is not divisible by stride {stride[0]} and patch size {patch_size[0]}"
    assert (W - patch_size[1]) % stride[1] == 0, f"Width {W} is not divisible by stride {stride[1]} and patch size {patch_size[1]}"

    # Calculate the number of windows along height and width
    num_patches_h = (H - patch_size[0]) // stride[0] + 1
    num_patches_w = (W - patch_size[1]) // stride[1] + 1

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[
                :,
                i*stride[0]:i*stride[0]+patch_size[0],
                j*stride[1]:j*stride[1]+patch_size[1]
            ]
            patches.append(patch)

    return torch.stack(patches)


def merge_grid(
    patches: torch.Tensor,
    image_size: Tuple[int, int],
    stride: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge patches into an image.

    Args:
        patches (torch.Tensor): Patches tensor of shape (num_patches, C, patch_size[0], patch_size[1]).
        image_size (Tuple[int, int]): Image size.
        stride (Tuple[int, int]): Stride of the patch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            Image tensor of shape (C, image_size[0], image_size[1]) and
            count tensor of shape (C, image_size[0], image_size[1]).
    """
    device = patches.device
    num_patches, C, H, W = patches.shape

    # Check if the image size is divisible by stride and patch size
    assert (image_size[0] - H) % stride[0] == 0, f"Height {image_size[0]} is not divisible by stride {stride[0]} and patch size {H}"
    assert (image_size[1] - W) % stride[1] == 0, f"Width {image_size[1]} is not divisible by stride {stride[1]} and patch size {W}"

    # Calculate the number of windows along height and width
    num_patches_h = (image_size[0] - H) // stride[0] + 1
    num_patches_w = (image_size[1] - W) // stride[1] + 1

    image = torch.zeros((C, image_size[0], image_size[1]), device=device)
    count = torch.zeros((C, image_size[0], image_size[1]), device=device)
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            image[
                :,
                i*stride[0]:i*stride[0]+H,
                j*stride[1]:j*stride[1]+W
            ] += patches[i*num_patches_w+j]
            count[
                :,
                i*stride[0]:i*stride[0]+H,
                j*stride[1]:j*stride[1]+W
            ] += 1

    return image, count


class DDPMSchedulerAdjusted(DDPMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        if not return_dict:
            return (pred_prev_sample,)

        output = DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
        output.variance = variance
        return output


class GVCFDiffusionMixin:
    """
    Base mixin for diffusion models implements step functions.
    """
    def generate_distance_map(self, size, batch_size, device):
        center = size // 2
        y, x = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing='ij'
        )
        distance_map = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        max_distance = torch.sqrt(torch.tensor(2 * (center ** 2)))
        normalized_distance_map = 1 - (distance_map / max_distance)
        normalized_distance_map = normalized_distance_map.unsqueeze(0).unsqueeze(0)
        normalized_distance_map = torch.repeat_interleave(
            normalized_distance_map, batch_size, dim=0)
        return normalized_distance_map

    def _vcf_step(self, epsilon_pred_patches, latent_patches, t, latent, stride):
        """
        Variance-Corrected Fusion (VCF) for SDE samplers such as DDPM.
        1. step on each epsilon patch
        2. merge and average, use adjusted avg.
        """
        image_size = latent.shape[1:]

        if isinstance(self.scheduler, DDPMScheduler):
            output = DDPMSchedulerAdjusted.step(
                self.scheduler, epsilon_pred_patches, t, latent_patches)
        else:
            raise ValueError(f'Scheduler {type(self.scheduler)} not supported.')

        latent_patches = output.prev_sample
        variance_patches = output.variance

        latent_mu, _ = merge_grid(latent_patches, image_size, stride)
        latent, count = merge_grid(latent_patches + variance_patches, image_size, stride)

        latent = torch.where(
            count > 0,
            latent / torch.sqrt(count) + (1 - torch.sqrt(count)) * (latent_mu / count),
            latent
        )

        return latent

    def _gvcf_step(self, epsilon_pred_patches, latent_patches, t, latent, stride):
        """
        Guided Variance-Corrected Fusion (GVCF) for SDE samplers such as DDPM.
        1. step on each epsilon patch
        2. merge and average with guidance, use adjusted avg.
        """
        image_size = latent.shape[1:]

        if isinstance(self.scheduler, DDPMScheduler):
            output = DDPMSchedulerAdjusted.step(
                self.scheduler, epsilon_pred_patches, t, latent_patches)
        else:
            raise ValueError(f'Scheduler {type(self.scheduler)} not supported.')

        latent_patches = output.prev_sample
        variance_patches = output.variance

        guidance = self.generate_distance_map(
            latent_patches.shape[2], latent_patches.shape[0], latent_patches.device
        )

        guidance_count, _ = merge_grid(guidance, image_size, stride)
        guidance_square_count, _ = merge_grid(guidance ** 2, image_size, stride)

        latent_mu, _ = merge_grid(latent_patches * guidance, image_size, stride)
        latent_mu = torch.where(guidance_count > 0, latent_mu / guidance_count, latent_mu)
        latent, _ = merge_grid((latent_patches + variance_patches) * guidance, image_size, stride)
        latent = torch.where(guidance_count > 0, latent / guidance_count, latent)

        latent = torch.where(
            guidance_count > 0,
            latent * guidance_count / torch.sqrt(guidance_square_count) + \
            (1 - guidance_count / torch.sqrt(guidance_square_count)) * latent_mu,
            latent
        )

        return latent

    def _gf_step(self, epsilon_pred_patches, latent_patches, t, latent, stride):
        """
        Guided Fusion (GF) for ODE samplers such as DDIM.
        1. step on each epsilon patch
        2. merge and average with the guidance
        """
        image_size = latent.shape[1:]

        latent_patches = self.scheduler.step(
            epsilon_pred_patches, t, latent_patches)['prev_sample']
        guidance = self.generate_distance_map(
            latent_patches.shape[2], latent_patches.shape[0], latent_patches.device
        )

        latent, _ = merge_grid(latent_patches * guidance, image_size, stride)
        gudiance_count, _ = merge_grid(guidance, image_size, stride)
        return torch.where(gudiance_count > 0, latent / gudiance_count, latent)
