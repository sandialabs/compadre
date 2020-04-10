# should be moved into metrics folder which is inside of canga folder
CANGA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga
METRICS_FOLDER=/ascldap/users/pakuber/Compadre/compadre/test_data/grids/canga/metrics
DATA_FOLDER=/ascldap/users/pakuber/Compadre/compadre/build/examples

# compute metrics
FIELDNAME="AnalyticalFun1"
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM8_CS-r16_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM8_ICOD-r16_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_16.g-ICOD_16.g --dimension 1 --output $METRICS_FOLDER/out_0.csv > $METRICS_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM8_CS-r32_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM8_ICOD-r32_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_32.g-ICOD_32.g --dimension 1 --output $METRICS_FOLDER/out_1.csv >> $METRICS_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM8_CS-r64_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM8_ICOD-r64_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_64.g-ICOD_64.g --dimension 1 --output $METRICS_FOLDER/out_2.csv >> $METRICS_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM8_CS-r128_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM8_ICOD-r128_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_128.g-ICOD_128.g --dimension 1 --output $METRICS_FOLDER/out_3.csv >> $METRICS_FOLDER/out.txt
python $METRICS_FOLDER/CANGAMetricsDriver.py --ss $METRICS_FOLDER/CS/sample_NM8_CS-r256_TPW_CFR_TPO_A1_A2.nc --st $METRICS_FOLDER/ICOD/sample_NM8_ICOD-r256_TPW_CFR_TPO_A1_A2.nc --smc 1 --tmc 1 --field $FIELDNAME  --data $DATA_FOLDER/CS_256.g-ICOD_256.g --dimension 1 --output $METRICS_FOLDER/out_4.csv >> $METRICS_FOLDER/out.txt

# append to one file
NAMES=""
for i in {0..4}; do NAMES="$NAMES $METRICS_FOLDER/out_${i}_${FIELDNAME}.csv"; done;
echo $NAMES
python $CANGA_FOLDER/get_rates.py $NAMES --target $METRICS_FOLDER/out.csv --norm GL2
