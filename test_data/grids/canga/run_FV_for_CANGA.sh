#!/bin/bash
trap "exit" INT TERM    # Convert INT and TERM to EXIT
trap "kill 0" EXIT      # Kill all children if we receive EXIT

# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
#DATA_FOLDER=/gpfs1/pakuber/CANGA
EXE_FOLDER=/ascldap/users/pakuber/Compadre/compadre/build/examples

#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_test/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_test/p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_test/p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_test/p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_test/p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/gc2_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/gc2_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/gc2_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/gc2_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/gc2_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/lbp_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/lbp_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/lbp_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/lbp_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD_rerun/lbp_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=true; done; done;

#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false; done; done;


#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false; done; done;

#DATA_FOLDER=/gpfs1/pakuber/CANGA/CANGA/CS_ICOD/lbp_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CANGA/CS_ICOD/lbp_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CANGA/CS_ICOD/lbp_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CANGA/CS_ICOD/lbp_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=true; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CANGA/CS_ICOD/lbp_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=true; done; done;

# haven't run correctly yet


#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc3_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc3_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc3_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc2_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/CS_ICOD/gc3_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CS --mesh-1=$i --mesh-2-type=CVT --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false; done; done;




#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;

#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/gc_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/gc_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/gc_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/gc_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/gc_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;

#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/lbp_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/lbp_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/lbp_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/lbp_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/ICOD_RLL/lbp_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=CVT --mesh-1=$i --mesh-2-type=RLL --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;

#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/gc_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/gc_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/gc_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/gc_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/gc_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;
#
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/lbp_p1
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/lbp_p2
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=2 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/lbp_p3
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=3 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/lbp_p4
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=4 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;
#DATA_FOLDER=/gpfs1/pakuber/CANGA/RLL_CS/lbp_p5
#for i in {0..4}; do for j in {0..4}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="CAAS" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=5 --save-every=10 --preserve-local-bounds=true --batches=1000; done; done;

DATA_FOLDER=/gpfs1/pakuber/CANGA/blah/p1
for i in {0..0}; do for j in {0..0}; do python $CANGA_FOLDER/remap_single_call.py --exe-folder-absolute=$EXE_FOLDER --canga-folder-absolute=$CANGA_FOLDER/NM16 --output-folder-absolute=$DATA_FOLDER --total-iterations=1000 --optimization="NONE" --mesh-1-type=RLL --mesh-1=$i --mesh-2-type=CS --mesh-2=$j --start-step=1 --porder=1 --save-every=10 --preserve-local-bounds=false --batches=1000; done; done;

# Find child processes and wait for them to finish so this script doesn't
# exit before the children do (otherwise our trap will kill them)
for job in $(jobs -p); do
    wait $job
done
