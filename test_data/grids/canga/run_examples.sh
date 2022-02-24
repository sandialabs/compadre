# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
EXE_FOLDER=/ascldap/users/pakuber/Compadre/compadre/build/examples
#DATA_FOLDER=/scratch/pakuber/test
DATA_FOLDER=/gpfs/pakuber/TEST
#DATA_FOLDER=/gpfs1/pakuber/CANGA

#for i in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=2 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$i --start-step=1 --porder=4 --save-every=1 --preserve-local-bounds=false --batches=1000; done;
for i in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=2 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$i --start-step=1 --porder=4 --save-every=1 --preserve-local-bounds=false --batches=1000; done;
#for i in {0..2}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=2 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CVT --mesh-2=$i --start-step=1 --porder=4 --save-every=1 --preserve-local-bounds=false --batches=1000; done;
