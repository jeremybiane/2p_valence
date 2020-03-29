#!/bin/bash

MOUSE=$1
NOTEBOOK=$2
DEST_MOUSE=$3


cp ../results/$MOUSE/notebooks/$NOTEBOOK.ipynb ../results/$DEST_MOUSE/notebooks;
sh exec_nb.sh ../results/$DEST_MOUSE/notebooks/$NOTEBOOK
