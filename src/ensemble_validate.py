import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from src.model import ImageClassifier
from src.datamodule import TreeDataModule
import yaml
from pathlib import Path
import torchvision.transforms as transforms
import argparse

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_backbone_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        backbone = checkpoint['hyper_parameters']['backbone']
        if 'vit' in backbone.lower():
            return load_config('configs/model/vit.yaml')
        elif 'efficient' in backbone.lower():
            return load_config('configs/model/efficientnet.yaml')
        else:
            return load_config('configs/model/resnet18.yaml')
    except:
        return load_config('configs/model/resnet18.yaml')

def ensemble_predict(models, batch, device, use_tta=False):
    predictions = []
    for model in models:
        model.eval()
        
        if use_tta:
            batch_preds = []
            for tta in model.tta_transforms:
                augmented_images = torch.stack([
                    tta(transforms.ToPILImage()(img.cpu())) 
                    for img in batch
                ]).to(device)
                
                with torch.no_grad():
                    pred = model(augmented_images)
                    batch_preds.append(torch.softmax(pred, dim=1))
            model_pred = torch.stack(batch_preds).mean(dim=0)
        else:
            with torch.no_grad():
                pred = model(batch)
                model_pred = torch.softmax(pred, dim=1)
        
        predictions.append(model_pred)
    
    # Усредняем предсказания всех моделей
    return torch.stack(predictions).mean(dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tta', action='store_true', help='Использовать Test Time Augmentation')
    args = parser.parse_args()

    config = load_config('configs/config.yaml')
    data_config = load_config('configs/datamodule/default.yaml')

    datamodule = TreeDataModule(
        data_dir=data_config['data_dir'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers']
    )
    datamodule.setup()

    models = []
    checkpoint_dir = 'ensemble'
    for checkpoint_file in os.listdir(checkpoint_dir):
        if checkpoint_file.endswith('.ckpt'):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            model_config = get_backbone_from_checkpoint(checkpoint_path)
            
            model = ImageClassifier.load_from_checkpoint(
                checkpoint_path,
                backbone=model_config['backbone'],
                num_classes=model_config['num_classes'],
                learning_rate=model_config['learning_rate']
            )
            models.append(model)
            print(f"Loaded model from {checkpoint_file} with backbone {model_config['backbone']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models:
        model.to(device)

    correct = 0
    total = 0
    val_loader = datamodule.val_dataloader()

    print(f"\nStarting validation{' with TTA' if args.use_tta else ''}...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            
            ensemble_preds = ensemble_predict(models, images, device, use_tta=args.use_tta)
            _, predicted = torch.max(ensemble_preds, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx}/{len(val_loader)} batches")

    accuracy = 100 * correct / total
    print(f"\nEnsemble Validation Accuracy{' with TTA' if args.use_tta else ''}: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 