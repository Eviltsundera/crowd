import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
from datetime import datetime

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    wandb_logger = WandbLogger(project=cfg.project_name)
    
    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    run_checkpoint_dir = os.path.join(cfg.checkpoint_dir, timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=run_checkpoint_dir,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_loss:.2f}",
                save_on_train_epoch_end=False,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=run_checkpoint_dir,
                monitor=None,
                save_top_k=-1,
                every_n_epochs=1,
                filename="checkpoint-{epoch:02d}",
                save_on_train_epoch_end=True,
            ),
        ],
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train() 