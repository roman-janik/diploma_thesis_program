# Author: Roman Janík
# Script for creating NER Pero OCR LM configs. For every trained LM model there are 3 types of datasets
# (cc - CNEC + CHNEC, po - Pero OCR NER, ccpo - CNEC + CHNEC + Pero OCR NER) and 2 lr rate + epochs combinations
# (lr_3e5_10_epochs, lr_3e5_15_epochs).
#


import os

from yaml import safe_dump, safe_load

comment = """# Author: Roman Janík
# Experiment YAML configuration file
#
# Notes:
#  Paths are relative to training script working directory.
#  Datasets are preprocessed to Hugging Face arrow format

"""
exp_configs_ner_dir = "../exp_configs_ner"
exp_configs_mlm_dir = "../exp_configs_mlm"
lm_configs = ["m_8L_128H_new_t_12k", "m_8L_128H_new_t_26k", "m_8L_128H_new_t_52k", "m_8L_128H_old_t_52k",
              "m_4L_256H_new_t_12k", "m_4L_256H_new_t_26k", "m_4L_256H_new_t_52k", "m_4L_256H_old_t_52k",
              "m_4L_512H_new_t_12k", "m_4L_512H_new_t_26k", "m_4L_512H_new_t_52k", "m_4L_512H_old_t_52k",
              "m_6L_256H_new_t_12k", "m_6L_256H_new_t_26k", "m_6L_256H_new_t_52k", "m_6L_256H_old_t_52k",
              "m_8L_512H_new_t_12k", "m_8L_512H_new_t_26k", "m_8L_512H_new_t_52k", "m_8L_512H_old_t_52k"]
dts_types = ["_dts_cc_", "_dts_po_", "_dts_ccpo_"]
lr_rate_epochs = ["lr_3e5_10_epochs", "lr_3e5_15_epochs"]
cnec_dts = {'name': 'CNEC 2.0 CoNLL', 'desc': 'Czech Named Entity Corpus 2.0 CoNNL dataset. General-language Czech '
                                              'NER dataset.', 'path': '../../datasets/cnec2.0_extended'}
poner_dts = {"name": "PONER 1.0", "desc": "Pero OCR NER 1.0 dataset. Historic-language Czech OCR-sourced NER dataset.",
             "path": "../../datasets/poner1.0"}
chnec_dts = {'name': 'CHNEC 1.0', 'desc': 'Czech Historical Named Entity Corpus 1.0 dataset. Historic-language Czech '
                                          'NER dataset.', 'path': '../../datasets/chnec1.0'}
dtss = {"_dts_cc_": {"cnec": cnec_dts, "chnec": chnec_dts},
        "_dts_po_": {"poner": poner_dts},
        "_dts_ccpo_": {"cnec": cnec_dts, "chnec": chnec_dts, "poner": poner_dts}}
num_epochs = {"lr_3e5_10_epochs": 10, "lr_3e5_15_epochs": 15}
dts_str = {"_dts_cc_": "combined CNEC and CHNEC", "_dts_po_": "PONER", "_dts_ccpo_": "combined CNEC, CHNEC and PONER"}
batch_size = 32

print("Script for creating NER Pero OCR LM configs. For every trained LM model there are 3 types of datasets (cc - "
      "CNEC + CHNEC, PONER - Pero OCR NER, ccmy - CNEC + CHNEC + Pero OCR NER) and 2 lr rate + epochs combinations ("
      "lr_3e5_10_epochs, lr_3e5_15_epochs). "
      "Based on 'linear_lr_3e5_10_epochs' config, only different values are changed.")

with open(os.path.join(exp_configs_ner_dir, "linear_lr_3e5_10_epochs.yaml")) as f:
    content = safe_load(f)

for lm_config in lm_configs:
    with open(os.path.join(exp_configs_mlm_dir, f"{lm_config}.yaml")) as f:
        lm_train_config = safe_load(f)
    content["model"]["name"] = lm_train_config["models"]["trained_model"]["name"]
    content["model"]["desc"] = lm_train_config["models"]["trained_model"]["desc"]
    content["model"]["path"] = f"../resources/ml_models/{lm_config}"

    for dts in dts_types:
        content["datasets"] = dtss[dts]

        for lr_ep in lr_rate_epochs:
            content["training"]["num_train_epochs"] = num_epochs[lr_ep]
            content["training"]["batch_size"] = batch_size
            content["training"]["val_batch_size"] = batch_size
            desc = f"NER Pero OCR LM model. Learning rate set to 3e-5. Learning rate scheduler is linear, " \
                   f"training {num_epochs[lr_ep]} epochs on {dts_str[dts]}."

            content["desc"] = desc

            filename = "lm_" + lm_config + dts + lr_ep + ".yaml"
            with open(os.path.join(exp_configs_ner_dir, filename), "w", encoding="utf-8") as f:
                f.write(comment)
                safe_dump(content, f)
