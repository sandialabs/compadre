#!/bin/bash
# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
METRICS_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga/metrics
DIM=1

# compute metrics
DATA_FOLDER=/gpfs1/pakuber/CANGA/$1
FIELDNAME=$2
# append to one file
NAMES=""
for i in {0..2}; do NAMES="$NAMES $DATA_FOLDER/out_${i}_${FIELDNAME}.csv"; done;
echo $NAMES
python $CANGA_FOLDER/get_rates.py $NAMES --target $DATA_FOLDER/out.csv --norm GL2
