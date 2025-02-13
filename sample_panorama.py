import os
import sys
from diffusion import seed_everything, StableDiffusion
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--negative', type=str, default='')
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=3584)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num', type=int, default=1,
                        help='number of images to generate, each with a incrementing seed (seed + i)')
    parser.add_argument('--outdir', type=str, default='results/samples')
    parser.add_argument('--step-func', type=str,
                        choices=[
                            'gf', 'vcf', 'gvcf'
                        ],
                        default='gvcf')
    parser.add_argument('--slerp', type=float, default=0.)
    parser.add_argument('--scheduler', type=str, default='auto',
                        choices=['auto', 'ddpm', 'ddim'])
    parser.add_argument('--steps', type=int, default=0,
                        help='number of steps to run the diffusion for, 0 means use the default number of steps')
    parser.add_argument('--compile', action='store_true')
    opt = parser.parse_args()

    if opt.scheduler == 'auto':
        if opt.step_func == 'gf':
            opt.scheduler = 'ddim'
        else:
            opt.scheduler = 'ddpm'

    # print options with formatting
    print('[INFO] Options:')
    for k, v in vars(opt).items():
        print(f'{k}: {v}')

    image_size = (opt.H, opt.W)
    stride = (opt.stride, opt.stride)

    model = StableDiffusion(scheduler=opt.scheduler, compile=opt.compile)
    if opt.steps:
        model.num_inference_steps = opt.steps
    kwargs = {
        'stride': stride,
        'step_func': opt.step_func,
        'image_size': image_size,
        'slerp': opt.slerp,
        'negative_prompts': opt.negative
    }
    gen = lambda prompts: model.text2panorama(prompts=prompts, **kwargs)

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
        for i in tqdm(range(opt.num), desc=prompt):
            prompt_valid = prompt.replace(' ', '_').replace('/', '_').replace('\\', '_').replace("'", '_')[:50]
            seed = opt.seed + i
            seed_everything(seed)

            img = gen(prompt)

            outname = os.path.join(opt.outdir, prompt_valid + f'_{opt.step_func}_{seed:06}.png')
            os.makedirs(os.path.dirname(outname), exist_ok=True)

            img.save(outname)
