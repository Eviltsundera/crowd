import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import wandb
from pytorch_lightning.loggers import WandbLogger

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    wandb_logger = WandbLogger(project=cfg.project_name)
    
    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_loss:.2f}"
            ),
        ],
    )
    
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train() 