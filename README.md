# [GVCFDiffusion: Guided and Variance-Corrected Fusion with One-shot Style Alignment for Large-Content Image Generation](https://arxiv.org/abs/2412.12771)

# Introduction

Producing large images using small diffusion models is gaining increasing popularity, as the cost of training large models could be prohibitive. A common approach involves jointly generating a series of overlapped image patches and obtaining large images by merging adjacent patches. However, results from existing methods often exhibit obvious artifacts, e.g., seams and inconsistent objects and styles. To address the issues, we proposed Guided Fusion (GF), which mitigates the negative impact from distant image regions by applying a weighted average to the overlapping regions. Moreover, we proposed Variance-Corrected Fusion (VCF), which corrects data variance at post-averaging, generating more accurate fusion for the Denoising Diffusion Probabilistic Model. Furthermore, we proposed a one-shot Style Alignment (SA), which generates a coherent style for large images by adjusting the initial input noise without adding extra computational burden. Extensive experiments demonstrated that the proposed fusion methods improved the quality of the generated image significantly. As a plug-and-play module, the proposed method can be widely applied to enhance other fusion-based methods for large image generation.

## Get Started

Prerequisites:
- Python 3.7 or higher

Install dependencies:
```bash
pip install -r requirements.txt
```

## Sample Panorama Images

```bash
python sample_panorama.py --H 512 --W 3584 --step-func {step func} --stride {stride} --slerp {alpha} --outdir {output directory} --prompt "text prompt"
# step-func: gf, vcf, gvcf 
# scheduler: if scheduler is not specified, the default scheduler is set
#    gf: ddim; vcf, gvcf: ddpm;
# stride: controls the overlap between patches
# slerp: 0.1, 0.2, ..., 1.0, can be ignored when slerp=0.0
```

Guided Fusion (`gf`) uses DDIM for fast sampling, Guided Variance-Corrected Fusion (`gvcf`) uses DDPM for high-quality sampling.

See sample_panorama.py script for more options

## Evaluation

### Step 1: Generate a reference dataset

```bash
python generate_reference.py --seed 0 --outdir {output directory} --scheduler {scheduler} --num 3500
# scheduler: ddim or ddpm
```

### Step 2: Generate a panorama dataset

```bash
python sample_panorama.py --H 512 --W 3584 --step-func {step func} --slerp {alpha} --outdir {output directory} --scheduler {scheduler}
# step-func: gf, vcf, gvcf 
# scheduler: if scheduler is not specified, the default scheduler is set
#    gf: ddim; vcf, gvcf: ddpm;
# slerp: 0.1, 0.2, ..., 1.0, can be ignored when slerp=0.0
```

### Step 3: Crop the panorama images to patches

```bash
cd eval  # under the eval directory
python crop_panorama.py {panorama directory}
# the cropped patches will be saved in the same directory with _patches suffix
# e.g.:
#   python crop_panorama.py ../results/panorama
#   the cropped patches will be saved in ../results/panorama_patches
```

### Step 4: Calculate scores (FID, KID, GIQA-QS/GIQA-DS, CLIP-Score)

```bash
# calculate all scores at once
cd eval  # under the eval directory
bash calculate_scores.sh {reference directory} {panorama patches directory}
# e.g.: bash calculate_scores.sh ../results/reference_ddpm ../results/panorama_patches
```

Look at the individual score calculation scripts for more details
- eval/eval_fid_kid.py
- eval/eval_giqa.py
- eval/eval_clip_score.py

We use [clean-fid](https://github.com/GaParmar/clean-fid) for calculating FID and KID scores. The files under `eval/GIQA` folder are copied from [GIQA](https://github.com/cientgu/GIQA) repo for calculating GIQA-QS/DS scores.

The FID, KID and CLIP scores are stored in separate txt files under the same directory as the panorama patches.

The GIQA scores are stored in two csv files under the same directory as the panorama patches. To acquire the GIQA QS and DS scores, calculate the mean of the respective columns.

```bash
# GIQA-QS
csvstat --mean -c 2 {reference vs. generated csv file}
# GIQA-DS
csvstat --mean -c 2 {generated vs. reference csv file}
```
