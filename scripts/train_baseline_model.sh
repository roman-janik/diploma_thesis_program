#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=2:gpu_cap=cuda75:gpu_mem=10gb:mem=20gb:scratch_ssd=10gb
#PBS -j oe

# Author: Roman Janík
# Metacentrum job shell script
##!!!!! IF YOU CHANGE WALLTIME DONT FORGET TO CHANGE TIMEOUT FOR TRAINING SCRIPT. !!!!!####
##!!!!! TRAINING SCRIPT TIMEOUT SHOULD BE AT LEAST 30 min. SHORTER THAN WALLTIME. !!!!!####

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/datasets/            # folder with datasets
RESPATH=$HOMEPATH/program/results/      # store results in this folder
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
mkdir datasets datasets/cnec2.0_extended datasets/chnec1.0
cp "$DATAPATH"/cnec2.0_extended/cnec2.0_extended.zip "$DATAPATH"/chnec1.0/chnec1.0.zip datasets
unzip -d datasets/cnec2.0_extended datasets/cnec2.0_extended.zip
unzip -d datasets/chnec1.0 datasets/chnec1.0.zip

# Download model
printf "Download model\n"
mkdir program/resources/
cp -r "$HOMEPATH"/program/resources/robeczech-base-pytorch program/resources/

# Prepare directory with results
printf "Prepare directory with results\n"
if [ ! -d "$HOMEPATH"/program/results/ ]; then # test if dir exists
  mkdir "$HOMEPATH"/program/ "$HOMEPATH"/program/results/
fi

# Prepare local directory with results
mkdir program/results program/results/model program/results/logs

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd program/diploma_thesis_program || exit 2
pip install --upgrade pip
TMPDIR=../../tmp pip install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt

# Start training
printf "\nStart training\n"
config_file="exp_configs/$config.yaml"
if [ -z "$modelpath" ]; then
    python train_baseline.py --config "$config_file"
    printf "Training exit code: %s\n" "$?"
else
#    mkdir ./logs
#    mkdir ./logs/latest_model
#    cp -r ${modelpath}/* ./logs/latest_model
    python script_na_trenovani
    printf "Training exit code: %s\n" "$?"
fi


# Save results
printf "\nSave results\n"
new_model_dir=$RESPATH/$(date +%Y-%m-%d-%H-%M)-${branch}-${stime}h
mkdir "$new_model_dir"
cp -r ../results/* "$new_model_dir"
cp "$config_file" "$new_model_dir"

# clean the SCRATCH directory
clean_scratch

