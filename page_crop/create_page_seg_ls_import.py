# Author: Roman Janík
# Script for creating an import file with URLs for Label Studio for page crop dataset. Result is Label Studio JSON
# format file. Script goes through image directory and adds .jpg page file to output json file.
#
# https://labelstud.io/guide/tasks.html
# https://labelstud.io/guide/storage.html#Tasks-with-local-storage-file-references
#

import json
import argparse
import os

from glob import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True,
                        help="Path to source directory with directory images.")
    parser.add_argument("-o", "--output_file", default="ls_import_file.json", help="Output file name.")
    parser.add_argument("-p", "--port", default="8081", help="Web server port for URLs.")
    parser.add_argument("-l", "--local_paths", dest="local_paths", action="store_true", default=False,
                        help="Use local paths instead of URLs.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    idx = 0
    result = []
    save_path = os.path.dirname(os.path.normpath(args.source_dir))

    print("Script for creating an import file with URLs for Label Studio. Result is Label Studio JSON format file. "
          "Script goes through image directory and adds .jpg page file to output json file. "
          "Output json file is saved to parent directory of source directory.")

    if args.local_paths:
        path_start = "/data/local-files/?d="
    else:
        path_start = "http://localhost:" + args.port + "/"

    print("Starting page processing...")

    for x_file in glob(args.source_dir + "/*.jpg"):
        file_name = os.path.basename(os.path.normpath(x_file))
        print("Processed page:    ", file_name)

        task = {
            "id": idx,
            "data": {
                "page_name": file_name.rstrip(".jpg"),
                "image": path_start + file_name
            }
        }
        result.append(task)
        idx += 1

    print("Saving result to: " + os.path.join(save_path, args.output_file))
    with open(os.path.join(save_path, args.output_file), "w", encoding='utf8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
