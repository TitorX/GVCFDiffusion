#!python3
import csv
import os
from subprocess import check_output

base = '../results/exp'
root = '../results/exp/reports_vs_ddpm'
ref = 'reference_ddpm'
output_file = f'{base}/full_reports_{ref}.csv'
header = ['Name', 'FID', 'KID', 'GIQA-QS', 'GIQA-DS', 'CLIP_Score']


with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for gen in os.listdir(base):
        if not gen.endswith('_patches'):
            continue

        if not os.path.exists(f"{root}/{ref}_vs_{gen}_fid_kid_scores.txt"):
            continue

        name = gen
        fid_kid_file = f"{root}/{ref}_vs_{gen}_fid_kid_scores.txt"
        giqa_qs_file = f"{root}/{ref}_vs_{gen}_giqa.csv"
        giqa_ds_file = f"{root}/{gen}_vs_{ref}_giqa.csv"
        clip_score_file = f"{root}/../clip_scores/{gen}_clip_scores.txt"

        with open(fid_kid_file, 'r') as f:
            fid_kid_lines = f.readlines()
            fid = fid_kid_lines[0].strip().split(': ')[1]
            kid = fid_kid_lines[1].strip().split(': ')[1]

        try:
            giqa_qs = check_output(['csvstat', '--mean', '-c', '2', giqa_qs_file]).decode().strip()
        except:
            giqa_qs = '-1'
        
        try:
            giqa_ds = check_output(['csvstat', '--mean', '-c', '2', giqa_ds_file]).decode().strip()
        except:
            giqa_ds = '-1'

        try:
            with open(clip_score_file, 'r') as f:
                clip_score = f.read().strip().split(': ')[1]
        except:
            clip_score = '-1'

        writer.writerow([name, fid, kid, giqa_qs, giqa_ds, clip_score])

print(f'Results have been written to {output_file}')
