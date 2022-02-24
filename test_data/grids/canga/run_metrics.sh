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
#DATA_FOLDER=/gpfs1/pakuber/CANGA/$1
DATA_FOLDER=/gpfs/pakuber/TEST/$1
FIELDNAME=$2
#"AnalyticalFun1"
#/gpfs1/pakuber/CANGA/CS_ICOD_test/p2
PROCS=8
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r16_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_16.g-ICOD_16.g --processes $PROCS --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt &
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_32.g-ICOD_32.g --processes $PROCS --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt &
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_64.g-ICOD_64.g --processes $PROCS --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt &
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_128.g-ICOD_128.g --processes $PROCS --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_256.g-ICOD_256.g --dimension $DIM --output $DATA_FOLDER/out_4.csv >> $DATA_FOLDER/out.txt &
## RLL->CSL
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r30-60_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/CS/sample_NM16_O10_CS-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/RLL_30-60.g-CS_16.g --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r90-180_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/CS/sample_NM16_O10_CS-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/RLL_90-180.g-CS_32.g --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r180-360_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/CS/sample_NM16_O10_CS-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/RLL_180-360.g-CS_64.g --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r360-720_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/CS/sample_NM16_O10_CS-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/RLL_360-720.g-CS_128.g --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r720-1440_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/CS/sample_NM16_O10_CS-r256_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/RLL_720-1440.g-CS_256.g --dimension $DIM --output $DATA_FOLDER/out_4.csv >> $DATA_FOLDER/out.txt &
# CVT->RLL
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r30-60_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_16.g-RLL_30-60.g --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r90-180_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_32.g-RLL_90-180.g --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r180-360_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_64.g-RLL_180-360.g --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r360-720_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_128.g-RLL_360-720.g --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/RLL/sample_NM16_O10_RLL-r720-1440_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_256.g-RLL_720-1440.g --dimension $DIM --output $DATA_FOLDER/out_4.csv >> $DATA_FOLDER/out.txt &
#FIELDNAME="CloudFraction"
#FIELDNAME="TotalPrecipWater"
#FIELDNAME="Topography"
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r16_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_16.g-ICOD_16.g --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_32.g-ICOD_32.g --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_64.g-ICOD_64.g --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_128.g-ICOD_128.g --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --includeGradientMetrics --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_256.g-ICOD_256.g --dimension $DIM --output $DATA_FOLDER/out_4.csv >> $DATA_FOLDER/out.txt &
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_32.g-ICOD_16.g --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_64.g-ICOD_32.g --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_128.g-ICOD_64.g --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_256.g-ICOD_128.g --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt

wait
# append to one file
NAMES=""
for i in {0..3}; do NAMES="$NAMES $DATA_FOLDER/out_${i}_${FIELDNAME}.csv"; done;
echo $NAMES
#python $CANGA_FOLDER/get_rates.py $NAMES --target $DATA_FOLDER/out.csv --norm GL2

# Find child processes and wait for them to finish so this script doesn't
# exit before the children do (otherwise our trap will kill them)
for job in $(jobs -p); do
    wait $job
done
