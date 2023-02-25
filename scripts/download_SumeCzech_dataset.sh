#!/bin/bash
#PBS -l select=1:ncpus=4:mem=20gb:scratch_ssd=20gb
#PBS -l walltime=3:00:00
#PBS -j oe

# Author: Roman Janík
# Metacentrum job shell script
# Download SumeCzech dataset
##!!!!! IF YOU CHANGE WALLTIME DONT FORGET TO CHANGE TIMEOUT FOR TRAINING SCRIPT. !!!!!####
##!!!!! TRAINING SCRIPT TIMEOUT SHOULD BE AT LEAST 30 min. SHORTER THAN WALLTIME. !!!!!####

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

# Download dataset script
printf "Download dataset script\n"
mkdir datasets datasets/sumeczech-1.0
cp "$DATAPATH"/sumeczech-1.0/sumeczech-1.0.zip  datasets
unzip -d datasets/sumeczech-1.0 datasets/sumeczech-1.0.zip
cp "$DATAPATH"/sumeczech-1.0/downloader.py "$DATAPATH"/sumeczech-1.0/downloader_extractor_utils.py datasets/sumeczech-1.0

# Copy already downloaded parts
cp "$DATAPATH"/sumeczech-1.0/*.jsonl datasets/sumeczech-1.0

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd datasets/sumeczech-1.0 || exit 2
pip install --upgrade pip
TMPDIR=../../tmp pip install -r requirements.txt

# Start downloading
printf "Start downloading\n"
python downloader.py --parallel 64 --download_start "$d_start" --download_end "$d_end"

# Save results
printf "\nSave results\n"

# Filter files
grep -xa '^{"abstract":.*' "sumeczech-1.0-dev.jsonl" > "sumeczech-1.0-dev_filtered.jsonl"
grep -xa '^{"abstract":.*' "sumeczech-1.0-oodtest.jsonl" > "sumeczech-1.0-oodtest_filtered.jsonl"
grep -xa '^{"abstract":.*' "sumeczech-1.0-test.jsonl" > "sumeczech-1.0-test_filtered.jsonl"
grep -xa '^{"abstract":.*' "sumeczech-1.0-train.jsonl" > "sumeczech-1.0-train_filtered.jsonl"

mv "sumeczech-1.0-dev_filtered.jsonl" "sumeczech-1.0-dev.jsonl"
mv "sumeczech-1.0-oodtest_filtered.jsonl" "sumeczech-1.0-oodtest.jsonl"
mv "sumeczech-1.0-test_filtered.jsonl" "sumeczech-1.0-test.jsonl"
mv "sumeczech-1.0-train_filtered.jsonl" "sumeczech-1.0-train.jsonl"

cat "sumeczech-1.0-dev.jsonl" >> "$DATAPATH"/sumeczech-1.0/"sumeczech-1.0-dev.jsonl"
cat "sumeczech-1.0-oodtest.jsonl" >> "$DATAPATH"/sumeczech-1.0/"sumeczech-1.0-oodtest.jsonl"
cat "sumeczech-1.0-test.jsonl" >> "$DATAPATH"/sumeczech-1.0/"sumeczech-1.0-test.jsonl"
cat "sumeczech-1.0-train.jsonl" >> "$DATAPATH"/sumeczech-1.0/"sumeczech-1.0-train.jsonl"

# clean the SCRATCH directory
clean_scratch
