# Author: Roman Janík
# Script for resizing all images in a folder. Only for JPEG format. PIL library is used.
#

import argparse
import os

from glob import glob
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True, help="Path to source directory with images.")
    parser.add_argument("-r", "--resize_ratio", required=True, type=float,
                        help="Resize ratio of image. Ratio 0.5 is half size, 2 is double size")
    parser.add_argument("-o", "--overwrite_image", default=False, action="store_true",
                        help="Overwrite existing images.")
    args = parser.parse_args()
    return args


def resize_aspect_fit(args):
    print("Image directory:\n", args.source_dir)

    for x_file in glob(args.source_dir + "/*.jpg"):
        image = Image.open(x_file)
        file_path, extension = os.path.splitext(os.path.normpath(x_file))

        new_image_height = int(image.size[0] / (1/args.resize_ratio))
        new_image_length = int(image.size[1] / (1/args.resize_ratio))

        image = image.resize((new_image_height, new_image_length), Image.LANCZOS)
        save_path = file_path + extension if args.overwrite_image else file_path + "_copy" + extension
        image.save(save_path, 'JPEG', optimize=True)


def main():
    args = get_args()
    print("Script for resizing JPEG images in a folder. If option -o --overwrite_image is set to True, images are "
          "overwritten! Otherwise a string '_copy' is appended to file name.")
    resize_aspect_fit(args)


if __name__ == '__main__':
    main()
