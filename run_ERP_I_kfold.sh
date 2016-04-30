NUM_FOLDS=12

TEST_ACC_STRING='testAvgClassAcctest'
values=( )
sum=0
SIM_TYPE='-1'

for fold_num in $(seq 1 $NUM_FOLDS)
do
	out_file="run_ERP_I_kfold.sh.sim$SIM_TYPE.$fold_num"
	th kfold_driver.lua -dropout_prob 0.2 -network_type fully_connected -num_folds 12 -max_presentations 1 -show_test -fold_num "$fold_num" -max_iterations 1000 -simulated $SIM_TYPE -ERP_I -l1_penalty 2 -l2_penalty 2 > $out_file
	echo "Completed $fold_num out of $NUM_FOLDS"
	# here we grab the maximum test value
	values[$fold_num]=$( cat "$out_file" | grep "$TEST_ACC_STRING" | cut -d':' -f2 | sort -nr | head -n1 )
	sum=$( echo "$sum+${values[$fold_num]}" | bc )
done

mean=$( echo "scale =3; $sum/$NUM_FOLDS" | bc )
echo "Beep..."
sleep 0.5
echo "Boop..."
sleep 0.5
echo "Beep..."
sleep 0.5
echo "========================"
echo "Calculations completed: "
echo "========================"
echo "Mean: $mean"
echo "========================"
echo "Max accuracies per fold"
( IFS=$'\n'; echo "${values[*]}" )
