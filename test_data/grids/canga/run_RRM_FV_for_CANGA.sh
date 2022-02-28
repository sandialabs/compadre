#!/bin/bash
trap "exit" INT TERM    # Convert INT and TERM to EXIT
trap "kill 0" EXIT      # Kill all children if we receive EXIT

# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
#DATA_FOLDER=/gpfs1/pakuber/CANGA
EXE_FOLDER=/ascldap/users/pakuber/Compadre/compadre/build/examples

#DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/p1
#for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/p2
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/p3
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/p4
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;

DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/gc2_p1
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/gc2_p2
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/gc2_p3
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/gc2_p4
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;

DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/lbp_p1
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/lbp_p2
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=true; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/lbp_p3
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=true; done; done;
DATA_FOLDER=/gpfs1/pakuber/CANGA/RRM_CS_ICOD/lbp_p4
for i in {0..2}; do for j in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM32 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RRMCS --mesh-1=$i --mesh-2-type=RRMCVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=true; done; done;

# Find child processes and wait for them to finish so this script doesn't
# exit before the children do (otherwise our trap will kill them)
for job in $(jobs -p); do
    wait $job
done
