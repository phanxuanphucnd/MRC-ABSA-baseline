export DATA_PATH=data/14lap
export PREPROCESS_DATA_PATH=data/14lap/preprocess
export PAIR_DATA_PATH=data/14lap
export SAVED_MODEL_PATH=models/

#TODO: Data_process
python data_process.py \
  --data_path $DATA_PATH \
  --output_path $PREPROCESS_DATA_PATH


#TODO: Make_data_dual
python make_data_dual.py \
  --data_path $PREPROCESS_DATA_PATH \
  --output_path $PREPROCESS_DATA_PATH

#TODO: Make_data_standard
python make_data_standard.py \
  --data_path $PAIR_DATA_PATH \
  --output_path $PREPROCESS_DATA_PATH

#TODO: Training
python main.py \
  --data_path $PREPROCESS_DATA_PATH \
  --epoch_num 40 \
  --mode train \
  --save_model_path $SAVED_MODEL_PATH