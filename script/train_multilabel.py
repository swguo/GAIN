import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from unit.config import config
from unit.dataset_multilabel import get_dataloaders
from unit.model import ContrastiveLearningModel

def train():
    wandb_logger = WandbLogger(project=config.PROJECT_NAME_MULTILABEL)
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=config.CHECKPOINT_DIR,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    train_loader, val_loader = get_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS, config.MAX_LENGTH)
    model = ContrastiveLearningModel(config.MODEL_NAME, learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback.best_model_path

if __name__ == "__main__":
    train()
