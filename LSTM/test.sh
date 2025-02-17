# Train model from data on each position
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.2_15/ --data_paths ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.2_30/ --data_paths ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.2_45/ --data_paths ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.2%_15/ --data_paths ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.25_30/ --data_paths ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.25_45/ --data_paths ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.3_15/ --data_paths ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.3_30/ --data_paths ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.3_45/ --data_paths ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.35_15/ --data_paths ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.35_30/ --data_paths ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/train.py --result_dir ./LSTM/results/0.35_45/ --data_paths ./motion_data_with_ground_truth_-0.35_45.csv



# Test all positions on best model
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_15/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_30/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_15/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_30/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.25_45/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_15/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_30/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.3_45/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_15/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_30/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv

python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
# python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.35_45/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv



# Train with all data, test on each position
python3 ./LSTM/train.py --result_dir ./LSTM/results/default/
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.2_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.2_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.2_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.25_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.25_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.25_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.3_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.3_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.3_45.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.35_15.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.35_30.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/default/ --test_data ./motion_data_with_ground_truth_-0.35_45.csv
python3 ./LSTM/plot_gait.py --result_dir ./LSTM/results/default/



# Test break positions on model trained with all data
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_femur_l.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_femur_r.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_tibia_r.csv
python3 ./LSTM/predict.py --model_dir ./LSTM/results/0.2_45/ --test_data ./motion_data_with_ground_truth_torso.csv



# Plot gait
python3 ./LSTM/plot_gait.py --result_dir ./LSTM/results/0.2_45/