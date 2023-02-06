# Author: Roman Janík
# Script for document page cropping. Uses trained segmentation model. Page header and footer are removed.
#

import transformers
import argparse
import os

from glob import glob
from PIL import Image, ImageDraw


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True,
                        help="Path to source directory with page images.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to output directory.")
    parser.add_argument("-m", "--model_path", required=True, help="Path to model for page segmentation.")
    parser.add_argument("-a", "--store_masks", dest="store_masks", action="store_true", default=False,
                        help="Store segmentation masks.")
    args = parser.parse_args()
    return args


def get_mask_rect_shape(mask: Image):
    seq_img = mask.getdata()
    mask_width = mask.size[0]
    rect_y_min = 0
    rect_y_max = 0
    page_region = False

    for i in range(0, len(seq_img), mask_width):
        if not page_region and seq_img[i] == 255:
            rect_y_min = (i // mask_width) + 5
            page_region = True
        if page_region and seq_img[i] == 0:
            rect_y_max = (i // mask_width) - 5
            break

    return 0, rect_y_min, mask_width, rect_y_max


def main():
    args = get_args()

    print("Script for document page cropping. Results are jpg "
          "format files. Script goes through document image directory and creates a new image without header"
          " and footer. Images are saved to chosen directory.")

    segmenter = transformers.pipeline("image-segmentation", model=args.model_path)

    print("Starting page image cropping...")

    for x_file in glob(os.path.normpath(args.source_dir) + "/*.jpg"):
        file_name = os.path.basename(os.path.normpath(x_file))
        print("Processed page:    ", file_name)
        with open(x_file, "rb") as img_f:
            old_img = Image.open(img_f)

            result = segmenter(old_img)
            page_mask = result[1]["mask"]
            # background = Image.new("RGBA", old_img.size, (0, 0, 0, 0))
            # new_img = Image.composite(old_img, background, page_mask)

            shape = get_mask_rect_shape(page_mask)
            new_img = old_img.crop(shape)

            # new_img = new_img.convert("RGB")
            new_img.save(os.path.join(args.output_dir, file_name))
            if args.store_masks:
                page_mask.save(os.path.join(args.output_dir, file_name.rstrip(".jpg") + "_mask.jpg"))

    print("Cropped images saved to:   " + args.output_dir)


if __name__ == '__main__':
    main()
