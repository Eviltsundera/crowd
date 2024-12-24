## Conda environment

```bash
conda create -n crowd python=3.10
conda activate crowd
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pytorch-lightning wandb hydra-core hydra-colorlog
conda install ipykernel pandas scikit-learn
pip install timm
pip install -e .
```

## Качаем данные

```bash
python ./scripts/download_data.py data/hw_3_markup_data.txt data/downloads/
```

## Разделяем данные на train и val

```bash
python ./scripts/split_train_val.py data/hw_3_markup_data.txt data/downloads/ data/split
```

## Запускаем обучение

## Репа должна выглядеть примерно так:

```
$ tree -d
.
├── checkpoints
├── configs
│   ├── datamodule
│   ├── model
│   └── trainer
├── data
│   ├── downloads
│   │   └── downloads
│   └── split
│       ├── train
│       │   ├── conifer
│       │   └── deciduous
│       └── val
│           ├── conifer
│           └── deciduous
├── notebooks
├── outputs
├── scripts
├── src
│   └── __pycache__
├── tree_classifier.egg-info
└── wandb
```
