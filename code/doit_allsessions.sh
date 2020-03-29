#!/bin/bash

# usage:
# sh doit_allsessions.sh ORIGIN_MOUSE ORIGIN_DAY ORIGIN_CONDITION ORIGIN_NOTEBOOK DESTINATION_MOUSE DESTINATION_DAY

for condition in `ls ../results/$5/$2/ | grep [p]`; do sh doit.sh $1 $2 $3 $4 $5 $6 $condition; done
