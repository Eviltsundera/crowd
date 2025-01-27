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
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class UrlImageDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = self.urls[idx]
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return self.transform(img)

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

def setup_device(args):
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Using device: {device}")
    return device

def load_models(checkpoint_dir, device):
    models = []
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
            model.to(device)
            if 'cuda' in str(device):
                model.cuda()
            models.append(model)
            print(f"Loaded model from {checkpoint_file} with backbone {model_config['backbone']}")
    return models

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
    
    return torch.stack(predictions).mean(dim=0)

def predict_urls(urls, models, device, batch_size=32, use_tta=False):
    dataset = UrlImageDataset(urls)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            batch = batch.to(device)
            predictions = ensemble_predict(models, batch, device, use_tta)
            probabilities, predicted = torch.max(predictions, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_confidences.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_confidences

def validate_models(models, datamodule, device, use_tta=False):
    correct = 0
    total = 0
    val_loader = datamodule.val_dataloader()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images, targets = images.to(device), targets.to(device)
            
            ensemble_preds = ensemble_predict(models, images, device, use_tta)
            _, predicted = torch.max(ensemble_preds, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tta', action='store_true', help='Использовать Test Time Augmentation')
    parser.add_argument('--input_file', type=str, help='Путь к файлу со списком URL')
    parser.add_argument('--output_file', type=str, help='Путь к выходному CSV файлу')
    parser.add_argument('--debug', action='store_true', help='Добавить столбец confidence в выходной файл')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    device = setup_device(args)
    
    if args.input_file:
        # Режим разметки
        if args.input_file.endswith('.xlsx'):
            df = pd.read_excel(args.input_file)
            urls = df['downloadUrl'].dropna().tolist()
        else:
            df = pd.read_csv(args.input_file)
            urls = df['downloadUrl'].dropna().tolist()
        
        models = load_models('ensemble', device)
        
        print(f"\nStarting prediction{' with TTA' if args.use_tta else ''}...")
        predictions, confidences = predict_urls(urls, models, device, use_tta=args.use_tta)
        
        df = pd.DataFrame({
            'downloadUrl': urls,
            'is_conifer': [bool(pred) for pred in predictions]
        })
        
        if args.debug:
            df['confidence'] = confidences
        
        output_file = args.output_file or 'predictions.csv'
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
    else:
        # Режим валидации
        data_config = load_config('configs/datamodule/default.yaml')
        datamodule = TreeDataModule(
            data_dir=data_config['data_dir'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers']
        )
        datamodule.setup()
        
        models = load_models('ensemble', device)
        
        print(f"\nStarting validation{' with TTA' if args.use_tta else ''}...")
        accuracy = validate_models(models, datamodule, device, use_tta=args.use_tta)
        print(f"\nEnsemble Validation Accuracy{' with TTA' if args.use_tta else ''}: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 