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
from ner_pipeline import NerPipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", required=True, help="Path to source directory with document directories.")
    parser.add_argument("-o", "--output_file", default="ls_import_file.json", help="Output file name.")
    parser.add_argument("-p", "--port", default="8081", help="Web server port for URLs.")
    parser.add_argument("-i", "--index", default=1, help="Start index for tasks/pages.")
    parser.add_argument("-d", "--drop_num", default=0, type=int,
                        help="Drop first x predictions. For non-document header entities.")
    parser.add_argument("-r", "--drop_rate", default=0., type=float,
                        help="Drop predictions with score less than drop rate. For prevention of bad annotations.")
    parser.add_argument("-l", "--local_paths", dest="local_paths", action="store_true", default=False,
                        help="Use local paths instead of URLs.")
    parser.add_argument("-t", "--tokenizer_path", dest="tokenizer_path", default="",
                        help="Path to tokenizer for generation of predictions.")
    parser.add_argument("-m", "--model_path", dest="model_path", default="",
                        help="Path to model for generation of predictions. If empty, no predictions are included.")
    parser.add_argument("-k", "--skip_docs", dest="skip_docs", action="store_true", default=False,
                        help="Skip documents listed in file 'skip_docs.txt' in parent directory of source directory.")
    args = parser.parse_args()
    return args


def get_prediction(ner_pipe: NerPipeline, task_id: int, task_file_path: str, drop_num: int, drop_rate: float):
    # mapping from model entities to Label Studio entities
    entity_types_map = {
        "p": "Personal names",
        "i": "Institutions",
        "g": "Geographical names",
        "t": "Time expressions",
        "o": "Artifact names/Objects"
    }
    span_predictions = []
    pred_id = 0
    task_text_chunks = []
    token_offset = 16
    start_chunk_pos = 0
    chunk_pos_offsets = [0]
    search_num_ent = 8
    m_max_length = ner_pipe.tokenizer.model_max_length

    with open(task_file_path, encoding="utf-8") as task_file_f:
        task_text = task_file_f.read()

        # check if file is empty
        if not task_text:
            # raise Exception("Page has no text, file is empty!")
            return {
                "model_version": ner_pipe.model_version,
                "score": 0.5,
                "result": []
            }

        tok_task_text = ner_pipe.tokenizer(task_text)
        tokens_len = max_token_len = len(tok_task_text["input_ids"])
        end_chunk_token_pos = max_token_len - 2 if max_token_len <= m_max_length else m_max_length - 1

        while tokens_len != 0:
            end_chunk_pos = tok_task_text.token_to_chars(end_chunk_token_pos).end
            task_text_chunk = task_text[start_chunk_pos:end_chunk_pos]
            task_text_chunks.append(task_text_chunk)

            tokens_len = 0 if tokens_len <= m_max_length else tokens_len - m_max_length + token_offset
            start_chunk_pos = tok_task_text.token_to_chars(end_chunk_token_pos - token_offset).end
            chunk_pos_offsets.append(start_chunk_pos)
            end_chunk_token_pos = max_token_len - 2 \
                if tokens_len <= m_max_length else end_chunk_token_pos + m_max_length - token_offset

        m_preds_chunks = ner_pipe(task_text_chunks)

        # loop prediction lists and adjust char position of entities relative to their chunk, skipping 1st (0th)
        for i in range(1, len(m_preds_chunks)):
            for j in range(len(m_preds_chunks[i])):
                m_preds_chunks[i][j]["start"] += chunk_pos_offsets[i]
                m_preds_chunks[i][j]["end"] += chunk_pos_offsets[i]

        # remove duplicate predictions (coming from offset spans)
        def are_duplicate_preds(pred_a, pred_b):
            equals = pred_a["entity_group"] == pred_b["entity_group"] and pred_a["word"] == pred_b["word"] \
                     and pred_a["end"] == pred_b["end"]
            pred_w_lower_score = None if not equals else "pred_a" if pred_a["score"] <= pred_b["score"] else "pred_b"
            return equals, pred_w_lower_score

        for i in range(len(m_preds_chunks)):
            if i + 1 < len(m_preds_chunks):
                j = 0
                j_end_idx = search_num_ent if search_num_ent <= len(m_preds_chunks[i + 1]) else len(
                    m_preds_chunks[i + 1])
                while j < j_end_idx:
                    k_start_idx = len(m_preds_chunks[i]) - 1
                    k_end_idx = k_start_idx - search_num_ent
                    k_end_idx = k_end_idx if k_end_idx > -1 else -1
                    for k in range(k_start_idx, k_end_idx, -1):
                        # print("j end idx: ", j_end_idx, "j: ", j, "k end idx: ", k_end_idx, "k: ", k)
                        preds_equals, pred_w_l_score = \
                            are_duplicate_preds(m_preds_chunks[i + 1][j], m_preds_chunks[i][k])
                        if preds_equals:
                            if pred_w_l_score == "pred_a":
                                m_preds_chunks[i + 1].pop(j)
                                j_end_idx = search_num_ent if search_num_ent <= len(m_preds_chunks[i + 1]) else len(
                                    m_preds_chunks[i + 1])
                            else:
                                m_preds_chunks[i].pop(k)
                            break
                    j += 1

        m_predictions = [item for sublist in m_preds_chunks for item in sublist]  # flatten result list

    for m_prediction in m_predictions:
        if m_prediction["score"] < drop_rate:
            continue
        span_prediction = {
            "id": str(task_id) + "_" + str(pred_id),
            "from_name": "ner",
            "to_name": "text_tag",
            "type": "labels",
            "value": {
                "start": m_prediction["start"],
                "end": m_prediction["end"],
                "score": float(m_prediction["score"]),
                "text": m_prediction["word"][1:] if m_prediction["word"][0] == " " else m_prediction["word"],
                "labels": [
                    entity_types_map[m_prediction["entity_group"]]
                ]
            }
        }
        span_predictions.append(span_prediction)
        pred_id += 1

    task_prediction = {
        "model_version": ner_pipe.model_version,
        "score": 0.5,
        "result": span_predictions[drop_num:]
    }

    return task_prediction


def main():
    args = get_args()
    idx = args.index
    doc_num_pages = 0
    result = []
    ner_pipeline = None
    skip_docs_list = []
    save_path = os.path.dirname(os.path.normpath(args.source_dir))

    print("Script for creating an import file with URLs for Label Studio. Result is Label Studio JSON format file. "
          "Script goes through subdirectories and adds .txt and .jpg page files to output json file. "
          "Output json file is saved to parent directory of source directory.")

    if args.local_paths:
        path_start = "/data/local-files/?d="
    else:
        path_start = "http://localhost:" + args.port + "/"

    if args.model_path != "":
        ner_pipeline = NerPipeline(args.tokenizer_path, args.model_path)

    if args.skip_docs:
        with open(os.path.join(save_path, "skip_docs.txt"), encoding="utf-8") as sfp:
            for line in sfp:
                skip_docs_list.append(line.rstrip())
        print("Documents to be skipped:\n", skip_docs_list)

    print("Starting documents processing...")

    for x_dir in natsorted(glob(os.path.normpath(args.source_dir) + "/*/", recursive=True)):
        document_name = os.path.basename(os.path.normpath(x_dir))

        # skip document if is it in skip document list
        if args.skip_docs and document_name in skip_docs_list:
            continue

        print("Processed document:    ", document_name)

        for x_file in natsorted(glob(x_dir + "/*.txt")):
            file_name = os.path.basename(os.path.normpath(x_file))
            # print("Processed page:    ", file_name)

            # check if both files exists
            if not (os.path.isfile(x_file) and os.path.isfile(x_file.replace(".txt", ".jpg"))):
                print("Warning: Either .txt or .jpg file '" + file_name.rstrip(".txt") + "' are missing! Skipping "
                                                                                         "file...")
                continue

            # add predictions, if model path is specified
            predictions = []
            if args.model_path != "":
                prediction = get_prediction(ner_pipeline, idx, x_file, args.drop_num, args.drop_rate)
                predictions.append(prediction)

            task = {
                "id": idx,
                "data": {
                    "document_name": document_name,
                    "page_name": file_name.rstrip(".txt"),
                    "text": path_start + document_name + "/" + file_name,
                    "image": path_start + document_name + "/" + file_name.replace(".txt", ".jpg")
                },
                "predictions": predictions
            }
            result.append(task)
            idx += 1
            doc_num_pages += 1
        print("Document finished, number of pages:    ", doc_num_pages)
        doc_num_pages = 0

    if not result:
        print("No document were processed, probably all were skipped! No result was saved.")
    else:
        print("Saving result to: " + os.path.join(save_path, args.output_file))
        with open(os.path.join(save_path, args.output_file), "w", encoding='utf8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
