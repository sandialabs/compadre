# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
DATA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/build/examples

cd $DATA_FOLDER
for i in {0..4}; do python $CANGA_FOLDER/remap_single_call.py 2 0 $i $i 1 4; done;
cd $CANGA_FOLDER
