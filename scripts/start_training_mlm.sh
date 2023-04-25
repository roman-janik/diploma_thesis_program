#!/bin/bash

# Author: Roman Janík
# Metacentrum start Masked Language Model job shell script
# Arguments:
# 1. git branch name
# 2. config name (no .yaml extension)
# 3. walltime in format HH:MM:SS

BRANCHNAME=$1
JTIMEOUT=$3
STIME=$(echo "$JTIMEOUT" | cut -d: -f1)
CONFIG=$2
FROM_STATE=$4
MIXED_PRECISION=$5

qsub -v branch="$BRANCHNAME",stime="$STIME",config="$CONFIG",from_state="$FROM_STATE",mixed_precision="$MIXED_PRECISION" -l walltime="$JTIMEOUT" ./train_mlm_model.sh
