#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=16:ngpus=2:gpu_cap=cuda75:mem=20gb:scratch_ssd=20gb
#PBS -j oe

##!!!!! IF YOU CHANGE WALLTIME DONT FORGET TO CHANGE TIMEOUT FOR TRAINING SCRIPT. !!!!!####
##!!!!! TRAINING SCRIPT TIMEOUT SHOULD BE AT LEAST 30 min. SHORTER THAN WALLTIME. !!!!!####

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/datasets/            # folder with datasets
RESPATH=$HOMEPATH/program/results/      # store results in this folder

cd $SCRATCHDIR

# clean the SCRATCH directory
clean_scratch

# Download the diploma_thesis_program repository
printf "Download the diploma_thesis_program repository\n"
mkdir program
cd program
git clone https://github.com/xjanik20/diploma_thesis_program
cd diploma_thesis_program
git checkout $branch
cd ../..

# Download dataset
printf "Download dataset\n"
mkdir datasets datasets/cnec2.0_extended datasets/chnec1.0
cp $DATAPATH/cnec2.0_extended/cnec2.0_extended.zip $DATAPATH/chnec1.0/chnec1.0.zip datasets
unzip -d datasets/cnec2.0_extended datasets/cnec2.0_extended.zip
unzip -d datasets/chnec1.0 datasets/chnec1.0.zip

# Prepare directory with results
printf "Prepare directory with results\n"
if [ ! -d "$HOMEPATH/program/results/" ]; then # test if dir exists
	mkdir $HOMEPATH/program/results/
	mkdir $HOMEPATH/program
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
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd program/diploma_thesis_program
pip install --upgrade pip
TMPDIR=../../tmp pip install --upgrade torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111 -r requirements.txt

# Start training.
if [ -z "$modelpath" ]; then
    python train_baseline.py --datasets_path "../../datasets" --model_path "../resources/robeczech-base-pytorch" --batch_size 16 --val_batch_size 16
else
    mkdir ./logs
    mkdir ./logs/latest_model
    cp -r ${modelpath}/* ./logs/latest_model
    python script_na_trenovani
fi

# Save model
new_model_dir=$RESPATH/$(date +%Y-%m-%d-%H)-${branch}-${stime}h
mkdir $new_model_dir
#cp -r logs $new_model_dir
cp -r ../results $new_model_dir

# clean the SCRATCH directory
clean_scratch
