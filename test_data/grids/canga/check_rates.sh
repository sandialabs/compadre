ORDER=2

CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
#DATA_FOLDER=/gpfs1/pakuber/CANGA/REMAP/GMLS
DATA_FOLDER=/gpfs1/pakuber/CANGA/REMAP/GMLS_just_p1_rerun/GMLS_plus_GC
declare -a rll_a_arr=("RLL16" "RLL32" "RLL64" "RLL128" "RLL256")
declare -a cs_a_arr=("CS16" "CS32" "CS64" "CS128" "CS256")
declare -a icod_a_arr=("ICOD16" "ICOD32" "ICOD64" "ICOD128" "ICOD256")

TYPE_1_A=( "${cs_a_arr[@]}" )
TYPE_2_A=( "${icod_a_arr[@]}" )
FIELDNAME="AnalyticalFun1"

for i in {0..4}; do NAMES="$NAMES $DATA_FOLDER/metrics_${TYPE_1_A[$i]}_${TYPE_2_A[$i]}_O${ORDER}_${FIELDNAME}.csv"; done;
echo $NAMES
python $CANGA_FOLDER/get_rates.py $NAMES --norm GL2
