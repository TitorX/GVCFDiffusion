import os
import torch
import argparse
from diffusion import StableDiffusion
from tqdm import tqdm


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--sd_version', type=str, default='2.0',
                        choices=['1.5', '2.0'],
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default='ddim')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--outdir', type=str, default='reference',)
    parser.add_argument('--num', type=int, default=500,
                        help='number of images to generate')
    opt = parser.parse_args()

    sd = StableDiffusion(sd_version=opt.sd_version,
                         scheduler=opt.scheduler,
                         compile=True)

    if opt.prompt:
        prompts = [opt.prompt]
    else:
        prompts = [
            'A photo of lush forest with a babbling brook',
            'An illustration of a beach in La La Land style',
            'Silhouette wallpaper of a dreamy scene with shooting stars',
            'A beach with palm trees',
            'A film photo of a beachside street under the sunset',
            'A photo of a city skyline at night',
            'Natural landscape in anime style illustration',
            'A photo of a snowy mountain peak with skiers',
            'A photo of a mountain range at twilight',
            'Cartoon panorama of spring summer beautiful nature'
        ]

    for prompt in prompts:
        prompt_valid = prompt.replace(' ', '_').replace('/', '_').replace('\\', '_').replace("'", '_')[:50]
        # use tqdm for progress bar
        pbar = tqdm(total=opt.num, desc=prompt)
        batch_idx = 0
        remain = opt.num
        seed = opt.seed
        while remain > 0:
            batch_size = min(remain, opt.batch_size)
            seed_everything(seed)
            imgs = sd.text2img(prompt, batch_size=batch_size)

            outdir = os.path.join(opt.outdir)
            os.makedirs(outdir, exist_ok=True)

            for i, img in enumerate(imgs):
                outname = os.path.join(
                    outdir,
                    prompt_valid + f'_{seed:06}_{batch_idx:06}_{i:06}.png'
                )
                img.save(outname)

            remain -= batch_size
            batch_idx += 1
            seed += 1
            pbar.update(batch_size)
