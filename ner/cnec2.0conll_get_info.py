# Author: Roman Janík
# Script for extracting named entity types and their counts from CNEC 2.0 CoNNL version dataset.
#

cnec_conll = "../../datasets/cnec2.0_extended/dev.conll"
cnec_stats = "../../datasets/cnec2.0_extended/stats.txt"

with open(cnec_conll, 'r', encoding="utf-8") as infile, open(cnec_stats, 'w') as outfile:
    hist = {}
    for line in infile:
        if line != '\n':
            entity_type = line.split()[-1]
        else:
            continue

        if entity_type != 'O':
            key = entity_type[-1]
            if entity_type[0] == 'B':
                hist[key] = hist.get(key, 0) + 1

    outfile.write("Statistics of CNEC 2.0 CoNNL dataset dev split:\n")
    for key, value in hist.items():
        outfile.write(f"{key}: {value}\n")
    outfile.write(f"Total entity types: {len(hist)}")
