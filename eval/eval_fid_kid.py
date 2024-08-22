import os
import argparse
from cleanfid import fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("real_dir", type=str, help="Path to the real images directory")
    parser.add_argument("fake_dir", type=str, help="Path to the fake images directory")
    args = parser.parse_args()

    save_path = \
        os.path.basename(args.real_dir) + \
        '_vs_' + os.path.basename(args.fake_dir) + \
        '_fid_kid_scores.txt'

    save_path = os.path.join(os.path.dirname(args.fake_dir), save_path)

    fid_value = fid.compute_fid(args.real_dir, args.fake_dir)
    kid_value = fid.compute_kid(args.real_dir, args.fake_dir)

    with open(save_path, 'w') as f:
        f.write(f"FID: {fid_value}\n")
        f.write(f"KID: {kid_value}\n")
