#!/bin/bash

MOUSE=$1
DAY=$2
CONDITION=$3
NOTEBOOK=$4
DEST_MOUSE=$5
DEST_DAY=$6

cp ../results/$MOUSE/$DAY/$CONDITION/notebooks/$NOTEBOOK.ipynb ../results/$DEST_MOUSE/$DEST_DAY/$CONDITION/notebooks;
sh exec_nb.sh ../results/$DEST_MOUSE/$DEST_DAY/$CONDITION/notebooks/$NOTEBOOK
