for i in {0..4}
do
for j in {0..4}
do
fo_name="$i-$j-P4-CAAS"
#fn_name="$i-$j-P2-CAAS"
#mv $fo_name $fn_name
out_name="$i-$j-P4.out"
#<< making directories
mkdir $fo_name
#<< running problems
echo $fo_name
python3 ../../test_data/grids/canga/remap_single_call.py 1000 2 $i $j 1 4 > $out_name
mv backward_* $fo_name
mv forward_* $fo_name
mv out* $fo_name
#<< moving
#mv $fo_name /gpfs1/pakuber/EXPERIMENTS/new_with_id
done
done

#nohup ./run_tests_and_move_files.sh 1>&2 | tee nohup.out &
