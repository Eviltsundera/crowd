import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import timm

class ImageClassifier(pl.LightningModule):
    def __init__(
        self, 
        backbone: str, 
        num_classes: int, 
        learning_rate: float,
        pretrained: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        self.train_acc = torchmetrics.Accuracy(
            task='binary' if num_classes == 2 else 'multiclass',
            num_classes=num_classes,
            top_k=1,
            threshold=0.5
        )
        self.val_acc = torchmetrics.Accuracy(
            task='binary' if num_classes == 2 else 'multiclass',
            num_classes=num_classes,
            top_k=1,
            threshold=0.5
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)