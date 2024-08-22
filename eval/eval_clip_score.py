import os
from glob import glob
import logging
import warnings
import numpy as np
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm


# disable warning
warnings.filterwarnings("ignore", module="huggingface_hub")
logging.getLogger("transformers").setLevel(logging.ERROR)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


seed_everything(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompts = [
    'A photo of a city skyline at night',
    'Natural landscape in anime style illustration',
    'A photo of a snowy mountain peak with skiers',
    'A photo of a mountain range at twilight',
    'Cartoon panorama of spring summer beautiful nature'
]
clean_prompt = lambda p: p.replace(' ', '_').replace('/', '_').replace('\\', '_').replace("'", '_')[:50]


def read_img(image_path):
    image = Image.open(image_path)
    image = T.ToTensor()(image)
    image = image.to(device)
    image = image * 255.0
    image = image.type(torch.int64)
    return image


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size for processing images')
    args = parser.parse_args()

    # remove the last slash if it exists
    if args.data_dir[-1] == '/':
        args.data_dir = args.data_dir[:-1]

    save_path = args.data_dir + '_clip_scores.txt'

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    metric = metric.to(device)

    for prompt in tqdm(prompts):
        prompt_valid = clean_prompt(prompt) + '*'
        prompt_files = glob(os.path.join(args.data_dir, prompt_valid))

        # batch for loop prompt files
        for i in tqdm(range(0, len(prompt_files), args.batch_size), leave=False):
            batch_files = prompt_files[i:i+args.batch_size]
            batch_prompts = [prompt] * len(batch_files)

            batch_images = [
                read_img(image_path)
                for image_path in batch_files
            ]
            batch_images = torch.stack(batch_images)
            metric(batch_images, batch_prompts)

    with open(save_path, 'w') as f:
        f.write(f"CLIP Score: {metric.compute().item()}")
