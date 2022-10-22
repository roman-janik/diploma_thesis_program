#!/bin/bash

BRANCHNAME=$1
JTIMEOUT=$2
SHOUR=$(echo $JTIMEOUT | cut -d: -f1)
STIME=$((SHOUR - 1))

qsub -v branch="$BRANCHNAME",stime="$STIME" -l walltime=$JTIMEOUT ./train_baseline_model.sh
