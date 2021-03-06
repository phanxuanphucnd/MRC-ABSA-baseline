# MRC for ABSA baseline

MRC approach for Aspect-based Sentiment Analysis (ABSA) without Opinion Extraction (OE)

**Paper:** [Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction](https://arxiv.org/abs/2103.07665)

**Dataset:** [https://github.com/xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data)

### Usage

- Prepare data:
```commandline
python data_process.py --data_path data/14lap

Arguments:
    --data_path :       Path to the dataset
```

```commandline
python make_data_dual --data_path data/14lap/preprocess

Arguments:
    --data_path :       Path to the dataset
```

```commandline
python make_data_standard --data_path data/14lab/pair --output_path ./data/14lap/preprocess

Arguments:
    --data_path  :      Path to the dataset
    --output_path:      Path to the output data      
```

- Training:
```commandline
python main.py \
    --data_path ./data/14lap/preprocess/ \
    --mode train \
    --model_type bert-base-uncased \
    --epoch_num 30 \
    --batch_size 4 \
    --learning_rate 1e-3
```
