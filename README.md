# diploma_thesis_program
 Program repository

This is a repository for master thesis "Document Information Extraction".  
Author: Roman Janík (xjanik20), 2023  
Faculty of Information Technology, Brno University of Technology

## Document Information Extraction
With development of digitization comes the need for historical document analysis. Named
Entity Recognition is an important task for Information extraction and Data mining. The
goal of this thesis is to develop a system for extraction of information from Czech historical
documents, such as newspapers, chronicles and registry books. An information extraction
system was designed, the input of which is scanned historical documents processed by the
OCR algorithm. The system is based on a modified RoBERTa model. The extraction of
information from Czech historical documents brings challenges in the form of the need for a
suitable corpus for historical Czech. The corpora Czech Named Entity Corpus (CNEC) and
Czech Historical Named Entity Corpus (CHNEC) were used to train the system, together
with my own created corpus. The system achieves 88.85 F1 score on CNEC and 87.19 F1
score on CHNEC, obtaining new state-of-the-art results.

## Content
This repository contains source scripts for training NER model, Masked Language model
and tokenizer. All auxiliary scripts for training are included. Scripts for creation and
preparation of datasets are also present. Additionally, necessary scripts for preparing,
training and using Page crop tool are included. Configuration files for NER and Masked
language model training are listed.

exp_configs_mlm - Masked language model training YAML configuration files  
exp_configs_models - RoBERTa model JSON configuration files  
exp_configs_ner - NER model training YAML configuration files  
mlm - scripts for Masked language model training  
ner - scripts for NER model training  
page_crop - Page crop tool  
README.md - Readme file  
requirements.txt - pip requirements file  
train_ml_model.py - Masked language model training  
train_ner_model.py - NER model training  
train_tokenizer.py - tokenizer training  

## Usage
#### Install
Python 3.10 is supported.
```bash
pip install torch==2.0.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt
```

#### Train ML model
```bash
python train_ml_model.py --config <cfg.yaml> --timeout <hours>
```

#### Train NER model
```bash
python train_ner_model.py --config <cfg.yaml> --results_csv <results.csv>
```

#### Train tokenizer
```bash
python train_tokenizer.py --new_tokenizer_dir <dir> --dataset_dirs <dir> --vocab_size <int>
```

### Citation
    JANÍK, Roman. Document Information Extraction. Brno, 2023. Master’s thesis. Brno
    University of Technology, Faculty of Information Technology. Supervisor Ing. Michal Hradiš, Ph.D.
