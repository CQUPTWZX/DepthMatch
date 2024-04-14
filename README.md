# Uncertainty-guided Multi-Expert Learning for Long-tailed Hierarchical Text Classification

## Requirements

* Python >= 3.6
* torch >= 1.6.0
* transformers == 4.2.1
* fairseq >= 0.10.0
* torch-geometric == 1.7.2
* torch-scatter == 2.0.8
* torch-sparse ==  0.6.12

## Preprocess

Please download the original dataset

## Train

```
usage: train.py [-h] [--lr LR] [--data {WebOfScience,nyt,rcv1}] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--warmup WARMUP] [--graph GRAPH]
                [--multi] [--lamb LAMB] [--thre THRE] [--layer LAYER] [--num_labels_list] [--seed SEED] [--wandb] [--ta] [--eta]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate.
  --data {WebOfScience,nyt,rcv1}
                        Dataset.
  --batch BATCH         Batch size.
  --early-stop EARLY_STOP
                        Epoch before early stop.
  --device DEVICE	cuda or cpu. Default: cuda
  --name NAME           A name for different runs.
  --update UPDATE       Gradient accumulate steps
  --warmup WARMUP       Warmup steps.
  --graph GRAPH         Whether use graph encoder. Default: True
  --multi               Whether the task is multi-label classification. Should keep default since all 
  						datasets are multi-label classifications. Default: True
  --lamb LAMB           Lambda
  --thre THRE           Threshold for keeping tokens. Denote as gamma in the paper.
  --layer LAYER         Label layer
  --num_labels_list     List of labels for each layer in the data set.
  --seed SEED           Random seed.
  --wandb               Use wandb for logging.
  --ta                  If prefix weight ≤ tau , the loss of expert m on the sample will be eliminated.
  --eta                 Eta is a temperature factor that adjusts the sensitivity of prefix weights.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

e.g. Train on `WebOfScience` with `batch=12, lambda=0.05, gamma=0.02`. Checkpoints will be in `checkpoints/WebOfScience-test/`.

```shell
python train.py --name test --batch 12 --data WebOfScience --lamb 0.05 --thre 0.02
```

### Reproducibility

The related parameter configuration has been published in the paper.

## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] [--name NAME] [--layer LAYER] [--num_labels_list] [--eta]

optional arguments:
  -h, --help            Show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --layer LAYER         Label layer
  --num_labels_list     List of labels for each layer in the data set.
  --eta                 Eta is a temperature factor that adjusts the sensitivity of prefix weights.
```

