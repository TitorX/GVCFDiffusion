'''
Crops panorama images to 512 x 512 images for evaluation.
'''
import os
import sys
import cv2
from tqdm import tqdm


def main(data_dir):
    save_dir = data_dir + '_patches'
    os.makedirs(save_dir, exist_ok=True)

    # Get all images in the data directory
    images = [
        f for f in os.listdir(data_dir) 
        if (not f.startswith('.') and '.png' in f or '.jpg' in f)
    ]

    # Crop panorama images to 512 x 512
    for image_name in tqdm(images):
        image = cv2.imread(os.path.join(data_dir, image_name))
        H, W, _ = image.shape

        h_crops = H // 512
        w_crops = W // 512

        for i in range(h_crops):
            for j in range(w_crops):
                crop = image[i*512:(i+1)*512, j*512:(j+1)*512]
                cv2.imwrite(
                    os.path.join(
                        save_dir,
                        f"{image_name[:-4]}_{i:04}_{j:04}.png"
                    ),
                    crop
                )
   
    print(f"[INFO] Cropped images saved to {save_dir}.")


if __name__=='__main__':
    main(sys.argv[1])
