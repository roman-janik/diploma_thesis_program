#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=2:gpu_cap=cuda70:mem=20gb:scratch_ssd=20gb
#PBS -j oe

##!!!!! IF YOU CHANGE WALLTIME DONT FORGET TO CHANGE TIMEOUT FOR TRAINING SCRIPT. !!!!!####
##!!!!! TRAINING SCRIPT TIMEOUT SHOULD BE AT LEAST 30 min. SHORTER THAN WALLTIME. !!!!!####

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/datasets/            #folder with datasets
RESPATH=$HOMEPATH/roberta_results/      #store results in this folder

cd $SCRATCHDIR

# clean the SCRATCH directory
clean_scratch

# Download the RoBERTa repository
printf "Download the RoBERTa repository\n"
git clone https://github.com/xjanik20/fairseq
cd fairseq-main
git checkout $branch
cd ..

# Download dataset
printf "Download dataset\n"
mkdir data
cp $DATAPATH/Czech_Named_Entity_Corpus_2.0.zip data
cp $DATAPATH/Czech_Historical_Named_Entity_Corpus_v_1.0.tgz data
unzip -d data Czech_Named_Entity_Corpus_2.0.zip
tar -xf data/Czech_Historical_Named_Entity_Corpus_v_1.0.tgz -C data

# Prepare directory with results
printf "Prepare directory with results\n"
if [ ! -d "$HOMEPATH/roberta_results/" ]; then # test if dir exists
	mkdir $HOMEPATH/roberta_results/
fi

printf "\-----------------------------------------------------------\n"
printf "JOB ID:             $PBS_JOBID\n"
printf "JOB NAME:           $PBS_JOBNAME\n"
printf "JOB SERVER NODE:    $PBS_SERVER\n"
printf "START TIME:         $(date +%Y-%m-%d-%H-%M)\n"
printf "GIT BRANCH:         $branch\n"
printf "\-----------------------------------------------------------\n"

# Prepare environment
printf "Prepare environment\n"
module load python36-modules-gcc
python3 -m venv env
source ./env/bin/activate
mkdir tmp
cd fairseq-main
pip install --upgrade pip
TMPDIR=../tmp pip install --upgrade torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

# Start training.
if [ -z "$modelpath" ]; then
    python3 script_na_trenovani
else
    mkdir ./logs
    mkdir ./logs/latest_model
    cp -r ${modelpath}/* ./logs/latest_model
    python3 script_na_trenovani
fi

# Save model
new_model_dir=$RESPATH/$(date +%Y-%m-%d-%H)-${branch}-${stime}h
mkdir $new_model_dir
cp -r logs $new_model_dir
cp -r results $new_model_dir

# clean the SCRATCH directory
clean_scratch
