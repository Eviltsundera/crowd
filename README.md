## Conda environment

```bash
conda create -n crowd python=3.10
conda activate crowd
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pytorch-lightning wandb hydra-core hydra-colorlog
conda install ipykernel pandas scikit-learn openpyxl
pip install timm requests
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

```bash
python src/train.py
```

## Валидация с использованием ансамбля моделей

1. Создайте директорию `ensemble` и поместите в неё чекпоинты моделей (`.ckpt` файлы)
2. Запустите валидацию одним из способов:

```bash
# Базовая валидация без TTA
python -m src.ensemble_validate

# Валидация с использованием Test Time Augmentation (TTA)
python -m src.ensemble_validate --use_tta
```

Скрипт автоматически определит архитектуру каждой модели в ансамбле и выполнит:
- Загрузку всех моделей из директории `ensemble`
- Валидацию на тестовом датасете
- Усреднение предсказаний всех моделей
- Вывод итоговой точности ансамбля

## Разметка новых данных

### Структура данных

В репозитории уже есть файл `data/model_check.xlsx` со списком URL для разметки. Вы можете использовать его как пример или создать свой файл в том же формате:

```
downloadUrl
https://example.com/image1.jpg
https://example.com/image2.jpg
...
```

### Запуск разметки

Для разметки изображений используйте:

```bash
# Разметка из подготовленного Excel файла
python -m src.ensemble_validate --input_file data/model_check.xlsx --output_file predictions.csv

# Разметка с использованием TTA
python -m src.ensemble_validate --input_file data/model_check.xlsx --output_file predictions.csv --use_tta

# Разметка с выводом уверенности модели
python -m src.ensemble_validate --input_file data/model_check.xlsx --output_file predictions.csv --debug
```

Если у вас есть список URL в другом формате, вы можете:
1. Конвертировать его в Excel формат
2. Использовать утилиту конвертации в txt:
```bash
python ./scripts/convert_xlsx_to_txt.py data/model_check.xlsx urls.txt
```

### Формат выходного файла

Результат сохраняется в CSV файл следующего формата:
```csv
downloadUrl,is_conifer[,confidence]
https://example.com/image1.jpg,TRUE[,0.95]
https://example.com/image2.jpg,FALSE[,0.87]
...
```
Столбец `confidence` добавляется только при использовании флага `--debug`.

## Структура проекта

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
│   ├── model_check.xlsx    # Файл со списком URL для разметки
│   └── split
│       ├── train
│       │   ├── conifer
│       │   └── deciduous
│       └── val
│           ├── conifer
│           └── deciduous
├── ensemble           # Директория для чекпоинтов ансамбля
├── notebooks
├── outputs
├── scripts
├── src
│   └── __pycache__
├── tree_classifier.egg-info
└── wandb
```
