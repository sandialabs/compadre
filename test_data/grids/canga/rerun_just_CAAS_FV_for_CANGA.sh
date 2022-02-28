#!/bin/bash
trap "exit" INT TERM    # Convert INT and TERM to EXIT
trap "kill 0" EXIT      # Kill all children if we receive EXIT

# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
DATA_FOLDER=/scratch/pakuber
EXE_FOLDER=/ascldap/users/pakuber/Compadre/compadre/build/examples

#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
##
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
#
DATA_FOLDER=/scratch/pakuber/CANGA/CS_ICOD/lbp_p1
for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true; done; done;

#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/gc_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/lbp_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
##
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/gc_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
##
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/lbp_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;

# Find child processes and wait for them to finish so this script doesn't
# exit before the children do (otherwise our trap will kill them)
for job in $(jobs -p); do
    wait $job
done
