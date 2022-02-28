#!/bin/bash
trap "exit" INT TERM    # Convert INT and TERM to EXIT
trap "kill 0" EXIT      # Kill all children if we receive EXIT

# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
METRICS_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga/metrics
#DATA_FOLDER=/gpfs1/pakuber/CANGA/p5
#DATA_FOLDER=/scratch/pakuber/test

DIM=101

# compute metrics
FIELDNAMES="CloudFraction,TotalPrecipWater,Topography,AnalyticalFun1,AnalyticalFun2"

declare -a rll_f_arr=("RLL-r30-60" "RLL-r90-180" "RLL-r180-360" "RLL-r360-720" "RLL-r720-1440")
declare -a cs_f_arr=("CS-r16" "CS-r32" "CS-r64" "CS-r128" "CS-r256")
declare -a icod_f_arr=("ICOD-r16" "ICOD-r32" "ICOD-r64" "ICOD-r128" "ICOD-r256")

declare -a rll_c_arr=("RLL_30-60" "RLL_90-180" "RLL_180-360" "RLL_360-720" "RLL_720-1440")
declare -a cs_c_arr=("CS_16" "CS_32" "CS_64" "CS_128" "CS_256")
declare -a icod_c_arr=("ICOD_16" "ICOD_32" "ICOD_64" "ICOD_128" "ICOD_256")

declare -a rll_a_arr=("RLL16" "RLL32" "RLL64" "RLL128" "RLL256")
declare -a cs_a_arr=("CS16" "CS32" "CS64" "CS128" "CS256")
declare -a icod_a_arr=("ICOD16" "ICOD32" "ICOD64" "ICOD128" "ICOD256")

OUTPUT_ROOT_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga/output_september_new_bounds_check

PROCS=8

function call_metrics(){
for k in {1..4}; 
do DATA_FOLDER=/gpfs1/pakuber/CANGA/${TYPE_1_NAME}_${TYPE_2_NAME}/$P_PRE$k OUTPUT_FOLDER=$OUTPUT_BASE_FOLDER/${POST_NAME}; 
    for i in {0..4}; 
        do for j in {0..4}; 
        do echo "python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/$TYPE_1_NAME/sample_NM16_O10_${TYPE_1_F[$i]}_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/$TYPE_2_NAME/sample_NM16_O10_${TYPE_2_F[$j]}_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAMES  --data $DATA_FOLDER/${TYPE_1_C[$i]}.g-${TYPE_2_C[$j]}.g --processes $PROCS --dimension $DIM --output $OUTPUT_FOLDER/metrics_${TYPE_1_A[$i]}_${TYPE_2_A[$j]}_O$((k+1)).csv > $OUTPUT_FOLDER/out_${POST_NAME}_${TYPE_1_A[$i]}_${TYPE_2_A[$j]}_p$k.txt"; 
                python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/$TYPE_1_NAME/sample_NM16_O10_${TYPE_1_F[$i]}_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/$TYPE_2_NAME/sample_NM16_O10_${TYPE_2_F[$j]}_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAMES  --data $DATA_FOLDER/${TYPE_1_C[$i]}.g-${TYPE_2_C[$j]}.g --processes $PROCS --dimension $DIM --output $OUTPUT_FOLDER/metrics_${TYPE_1_A[$i]}_${TYPE_2_A[$j]}_O$((k+1)).csv > $OUTPUT_FOLDER/out_${POST_NAME}_${TYPE_1_A[$i]}_${TYPE_2_A[$j]}_p$k.txt; 
                #python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/$TYPE_1_NAME/sample_NM16_O10_${TYPE_1_F[$i]}_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/$TYPE_2_NAME/sample_NM16_O10_${TYPE_2_F[$j]}_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAMES  --data $DATA_FOLDER/${TYPE_1_C[$i]}.g-${TYPE_2_C[$j]}.g --processes $PROCS --dimension $DIM --output $OUTPUT_FOLDER/metrics_${TYPE_1_A[$i]}_${TYPE_2_A[$j]}_O$((k+1)).csv > $OUTPUT_FOLDER/out_${POST_NAME}_${TYPE_1_A[$i]}_${TYPE_2_A[$j]}_p$k.txt; 
        done; 
    done; 
done;
}

mkdir $OUTPUT_ROOT_FOLDER/GMLS
mkdir $OUTPUT_ROOT_FOLDER/GMLS_plus_GC
mkdir $OUTPUT_ROOT_FOLDER/GMLS_plus_CAAS
OUTPUT_BASE_FOLDER=$OUTPUT_ROOT_FOLDER

# CS->ICOD
TYPE_1_NAME=CS
TYPE_2_NAME=ICOD
POST_NAME=GMLS
TYPE_1_F=( "${cs_f_arr[@]}" )
TYPE_2_F=( "${icod_f_arr[@]}" )
TYPE_1_C=( "${cs_c_arr[@]}" )
TYPE_2_C=( "${icod_c_arr[@]}" )
TYPE_1_A=( "${cs_a_arr[@]}" )
TYPE_2_A=( "${icod_a_arr[@]}" )

#POST_NAME=GMLS
#P_PRE=p
#call_metrics

POST_NAME=GMLS_plus_GC
P_PRE=gc_p
call_metrics

POST_NAME=GMLS_plus_CAAS
P_PRE=lbp_p
call_metrics

# ICOD->RLL
TYPE_1_NAME=ICOD
TYPE_2_NAME=RLL
POST_NAME=GMLS
TYPE_1_F=( "${icod_f_arr[@]}" )
TYPE_2_F=( "${rll_f_arr[@]}" )
TYPE_1_C=( "${icod_c_arr[@]}" )
TYPE_2_C=( "${rll_c_arr[@]}" )
TYPE_1_A=( "${icod_a_arr[@]}" )
TYPE_2_A=( "${rll_a_arr[@]}" )

POST_NAME=GMLS
P_PRE=p
call_metrics

POST_NAME=GMLS_plus_GC
P_PRE=gc_p
call_metrics

POST_NAME=GMLS_plus_CAAS
P_PRE=lbp_p
call_metrics

# RLL->CS
TYPE_1_NAME=RLL
TYPE_2_NAME=CS
POST_NAME=GMLS
TYPE_1_F=( "${rll_f_arr[@]}" )
TYPE_2_F=( "${cs_f_arr[@]}" )
TYPE_1_C=( "${rll_c_arr[@]}" )
TYPE_2_C=( "${cs_c_arr[@]}" )
TYPE_1_A=( "${rll_a_arr[@]}" )
TYPE_2_A=( "${cs_a_arr[@]}" )

POST_NAME=GMLS
P_PRE=p
call_metrics

POST_NAME=GMLS_plus_GC
P_PRE=gc_p
call_metrics

POST_NAME=GMLS_plus_CAAS
P_PRE=lbp_p
call_metrics






#wait
## append to one file
#NAMES=""
#FIELDNAME="TotalPrecipWater"
#for i in {0..3}; do NAMES="$NAMES $OUTPUT_FOLDER/out_${i}_${FIELDNAME}.csv"; done;
#echo $NAMES
#python $CANGA_FOLDER/get_rates.py $NAMES --target $OUTPUT_FOLDER/out.csv --norm GL2

# Find child processes and wait for them to finish so this script doesn't
# exit before the children do (otherwise our trap will kill them)
for job in $(jobs -p); do
    wait $job
done
