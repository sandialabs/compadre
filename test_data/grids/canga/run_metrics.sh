# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
METRICS_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga/metrics
DATA_FOLDER=/scratch/pakuber

DIM=3

# compute metrics
FIELDNAME="AnalyticalFun2"
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r16_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_16.g-ICOD_16.g --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_32.g-ICOD_32.g --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_64.g-ICOD_64.g --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_128.g-ICOD_128.g --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM16_O10_CS-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_256.g-ICOD_256.g --dimension $DIM --output $DATA_FOLDER/out_4.csv >> $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_32.g-ICOD_16.g --dimension $DIM --output $DATA_FOLDER/out_0.csv > $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_64.g-ICOD_32.g --dimension $DIM --output $DATA_FOLDER/out_1.csv >> $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_128.g-ICOD_64.g --dimension $DIM --output $DATA_FOLDER/out_2.csv >> $DATA_FOLDER/out.txt
#python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM16_O10_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/ICOD_256.g-ICOD_128.g --dimension $DIM --output $DATA_FOLDER/out_3.csv >> $DATA_FOLDER/out.txt

# append to one file
NAMES=""
for i in {0..3}; do NAMES="$NAMES $DATA_FOLDER/out_${i}_${FIELDNAME}.csv"; done;
echo $NAMES
python $CANGA_FOLDER/get_rates.py $NAMES --target $DATA_FOLDER/out.csv --norm GL2
