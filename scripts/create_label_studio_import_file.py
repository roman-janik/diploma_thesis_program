# Author: Roman Janík
# Script for creating an import file with URLs for Label Studio. Result is Label Studio JSON format file.
# Script goes through subdirectories and adds .txt and .jpg page files to output json file.
#
# https://labelstud.io/guide/tasks.html
# https://labelstud.io/guide/storage.html#Tasks-with-local-storage-file-references
#

import json
import argparse
import os

from glob import glob
from natsort import natsorted


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True, help="Path to source directory with document directories.")
    parser.add_argument("-o", "--output_file", default="ls_import_file.json", help="Output file name.")
    parser.add_argument("-p", "--port", default="8081", help="Web server port for URLs.")
    parser.add_argument("-l", "--local_paths", dest="local_paths", action="store_true", default=False,
                        help="Use local paths instead of URLs.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    path_start = ""
    idx = 0
    result = []

    print("Script for creating an import file with URLs for Label Studio. Result is Label Studio JSON format file. "
          "Script goes through subdirectories and adds .txt and .jpg page files to output json file.")

    if args.local_paths:
        path_start = "/data/local-files/?d="
    else:
        path_start = "http://localhost:" + args.port + "/"

    for x_dir in natsorted(glob(os.path.normpath(args.source_dir) + "*/", recursive=True)):
        document_name = os.path.basename(os.path.normpath(x_dir))

        for x_file in natsorted(glob(x_dir + "/*.jpg")):
            file_name = os.path.basename(os.path.normpath(x_file))

            # check if both files exists
            if not (os.path.isfile(x_file) and os.path.isfile(x_file.replace(".jpg", ".txt"))):
                print("Warning: Either .jpg or .txt file " + file_name.rstrip(".jpg") + "are missing! Skipping file...")
                continue

            task = {"id": idx,
                    "data": {
                        "document_name": document_name,
                        "page_name": file_name.rstrip(".jpg"),
                        "text": path_start + document_name + "/" + file_name.replace(".jpg", ".txt"),
                        "image": path_start + document_name + "/" + file_name
                    }}
            result.append(task)
            idx += 1

    save_path = os.path.dirname(os.path.normpath(args.source_dir))
    print("Saving result to: " + os.path.join(save_path, args.output_file))
    json_text = json.dumps(result, indent=2)
    with open(os.path.join(save_path, args.output_file), "w") as f:
        f.write(json_text)
