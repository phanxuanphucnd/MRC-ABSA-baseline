export PREPROCESS_DATA_PATH=data/14lap/preprocess
export SAVED_MODEL_PATH=models/14lap

#TODO: Training
python main.py \
  --data_path $PREPROCESS_DATA_PATH \
  --mode test \
  --save_model_path $SAVED_MODEL_PATH
