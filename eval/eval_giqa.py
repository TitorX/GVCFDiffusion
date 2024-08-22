import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference_dir', type=str)
    parser.add_argument('eval_dir', type=str)
    parser.add_argument('--K', type=int, default=3500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    reference_dir = args.reference_dir
    eval_dir = args.eval_dir
    gpu = args.gpu
    K = args.K
    batch_size = args.batch_size

    act_file = os.path.join(
        os.path.dirname(reference_dir),
        f'{os.path.basename(reference_dir)}_act.pickle'
    )

    if not os.path.exists(act_file):
        print('Computing activations...')
        os.system(f'python GIQA/write_act.py {reference_dir} --act_path {act_file} --batch_size {batch_size} --gpu {gpu}')
        print('Done.')

    output_file = os.path.join(
        os.path.dirname(eval_dir),
        f'{os.path.basename(reference_dir)}_vs_{os.path.basename(eval_dir)}_giqa.csv'
    )

    print('Computing GIQA...')
    os.system(f'python GIQA/knn_score.py {eval_dir} --act_path {act_file} --K {K} --output_file {output_file} --gpu {gpu}')
