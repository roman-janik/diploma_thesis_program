#!/bin/bash

# Author: Roman Janík
# Metacentrum start job shell script
# Arguments:
# 1. git branch name
# 2. config name (no .yaml extension)
# 3. walltime in format HH:MM:SS

BRANCHNAME=$1
JTIMEOUT=$3
SHOUR=$(echo $JTIMEOUT | cut -d: -f1)
STIME=$((SHOUR - 1))
CONFIG=$2

qsub -v branch="$BRANCHNAME",stime="$STIME",config=$CONFIG  -l walltime=$JTIMEOUT ./train_baseline_model.sh
