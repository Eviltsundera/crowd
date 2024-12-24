import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import torchmetrics

class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        
        # Загружаем предобученную модель
        self.model = getattr(models, model_name)(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Метрики
        self.train_acc = torchmetrics.Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits.softmax(dim=-1), y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits.softmax(dim=-1), y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)