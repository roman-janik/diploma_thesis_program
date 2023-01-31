#!/bin/bash

# Author: Roman Janík
# Metacentrum start job shell script
# Arguments:
# 1. git branch name
# 2. walltime in format HH:MM:SS

BRANCHNAME=$1
JTIMEOUT=$3
SHOUR=$(echo "$JTIMEOUT" | cut -d: -f1)
STIME=$((SHOUR - 1))


qsub -v branch="$BRANCHNAME",stime="$STIME" -l walltime="$JTIMEOUT" ./train_page_crop_model.sh
