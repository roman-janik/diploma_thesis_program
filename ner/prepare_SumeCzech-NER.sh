#!/bin/bash
#PBS -l select=1:ncpus=1:mem=20gb:scratch_ssd=20gb
#PBS -j oe

# Author: Roman Janík
# Metacentrum job shell script
# Prepare SumeCzech-NER dataset into Hugging Face dataset format
##!!!!! IF YOU CHANGE WALLTIME DONT FORGET TO CHANGE TIMEOUT FOR TRAINING SCRIPT. !!!!!####
##!!!!! TRAINING SCRIPT TIMEOUT SHOULD BE AT LEAST 30 min. SHORTER THAN WALLTIME. !!!!!####

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/datasets/            # folder with datasets
HOSTNAME=$(hostname -f)                 # hostname of local machine

printf "\-----------------------------------------------------------\n"
printf "JOB ID:             %s\n" "$PBS_JOBID"
printf "JOB NAME:           %s\n" "$PBS_JOBNAME"
printf "JOB SERVER NODE:    %s\n" "$HOSTNAME"
printf "START TIME:         %s\n" "$(date +%Y-%m-%d-%H-%M)"
#printf "GIT BRANCH:         $branch\n"
printf "\-----------------------------------------------------------\n"


cd "$SCRATCHDIR" || exit 2

# clean the SCRATCH directory
clean_scratch

# Clone the diploma_thesis_program repository
printf "Clone the diploma_thesis_program repository\n"
cp "$HOMEPATH"/.ssh/id_ed25519 "$HOMEPATH"/.ssh/known_hosts "$HOME"/.ssh
printf "Print content of .ssh dir\n"
ls -la "$HOME"/.ssh
mkdir program
cd program || exit 2
git clone git@github.com:xjanik20/diploma_thesis_program.git
if [ $? != 0 ]; then
  printf "Cloning diploma_thesis_program repository failed!\n"
  exit 1
fi
cd diploma_thesis_program || exit 2
git checkout "$branch"
cd ../..

# Download dataset
printf "Download dataset\n"
mkdir datasets datasets/sumeczech-1.0-ner datasets/sumeczech-1.0
cp "$DATAPATH"/sumeczech-1.0/* datasets/sumeczech-1.0
cd datasets/sumeczech-1.0-ner || exit 2
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3505{/sumeczech-1.0-ner-0.jsonl,/sumeczech-1.0-ner-1.jsonl,/sumeczech-1.0-ner-2.jsonl,/sumeczech-1.0-ner-3.jsonl}
cd ../..

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd program/diploma_thesis_program || exit 2
pip install --upgrade pip
TMPDIR=../../tmp pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt

printf "Start processing\n"
cd scripts || exit 2
python prepare_cnec_chnec_datasets.py


# Save results
printf "\nSave results\n"
rm ../../../datasets/sumeczech-1.0-ner/*.jsonl
cp -r ../../../datasets/sumeczech-1.0-ner/* "$DATAPATH"/sumeczech-1.0-ner/

# clean the SCRATCH directory
clean_scratch
