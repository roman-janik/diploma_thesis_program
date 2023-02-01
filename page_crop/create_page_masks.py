# Author: Roman Janík
# Script for creating page masks from Label Studio export XML file for page crop dataset. Results are png
# format files. Script goes through annotations directory and creates .png page mask files.
#

import argparse
import os

from parsel import Selector
from glob import glob
from PIL import Image, ImageDraw


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True,
                        help="Path to source directory with page xml annotations.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to output directory.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print("Script for creating page masks from Label Studio export XML file for page crop dataset. Results are png "
          "format files. Script goes through annotations directory and creates .png page mask files.")

    print("Starting page annotations processing...")

    for x_file in glob(args.source_dir + "/*.xml"):
        file_name = os.path.basename(os.path.normpath(x_file))
        print("Processed page:    ", file_name)

        with open(x_file, encoding="utf-8") as x_file_p:
            tree = Selector(x_file_p.read())

            # extract annotation values from XML file
            image_name = tree.xpath("//filename/text()").getall()
            mask_name = image_name[0].replace(".jpg", ".png")
            mask_width = int(tree.xpath("//size/width/text()").getall()[0])
            mask_height = int(tree.xpath("//size/height/text()").getall()[0])
            rect_x_min = 0
            rect_y_min = int(tree.xpath("//object/bndbox/ymin/text()").getall()[0])
            rect_x_max = mask_width
            rect_y_max = int(tree.xpath("//object/bndbox/ymax/text()").getall()[0])

            # create mask
            shape = [(rect_x_min, rect_y_min), (rect_x_max, rect_y_max)]
            img = Image.new("L", (mask_width, mask_height), color="black")
            img_draw = ImageDraw.Draw(img)
            img_draw.rectangle(shape, fill="#010101")

            img.save(os.path.join(args.output_dir, mask_name))

    print("Page mask saved to:   " + args.output_dir)


if __name__ == '__main__':
    main()
