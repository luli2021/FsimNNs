import os
import shutil

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from datasets.customized_dataset import CustomGraphDataset, DisplayDatasetInfo

from models.models import get_model
from utils.utils import get_logger
from utils.config import load_config



logger = get_logger('FsimNNTraining')


class GNNLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=5e-4, factor=0.5, patience=10):
        super(GNNLightning, self).__init__()
        self.model = model  # Accepts any pre-defined GNN model
        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.best_val_acc = 0.0
        


    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        acc = self.accuracy(out, batch.y, batch.train_mask)
        self.log("train_loss", loss, batch_size=batch.train_mask.sum().item())
        self.log("train_acc", acc, batch_size=batch.train_mask.sum().item())
        logger.info(f"[Epoch{self.current_epoch}/Iteration{batch_idx}]: loss = {loss.item()}, acc = {acc.item()}")

        optimizer = self.optimizers()
        for param_group in optimizer.param_groups:
            logger.info(f"Epoch {self.current_epoch+1}, Batch {batch_idx+1}, Learning Rate: {param_group['lr']:.6f}")

        return loss
    
    #def on_after_backward(self):                
    #    for name, param in self.named_parameters():
    #        if param.grad is not None:  # Avoid NoneType errors
    #            print(f"Layer: {name} | Gradient Norm: {param.grad.norm().item()}") 
    

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.cross_entropy(out[batch.val_mask], batch.y[batch.val_mask])
        acc = self.accuracy(out, batch.y, batch.val_mask)
        self.log("val_loss", loss, batch_size=batch.val_mask.sum().item())
        self.log("val_acc", acc, batch_size=batch.val_mask.sum().item())
        
    def on_validation_epoch_end(self):
        """Runs once at the end of the validation epoch."""
        # Get all logged values from `self.trainer.callback_metrics`
        avg_loss = self.trainer.callback_metrics["val_loss"]
        avg_acc = self.trainer.callback_metrics["val_acc"]
        logger.info(f'Validation finished. Loss: {avg_loss.item()}, Acc: {avg_acc.item()}')

        if avg_acc.item() > self.best_val_acc:
            self.best_val_acc = avg_acc.item()
            logger.info(f'New best validation accuracy: {self.best_val_acc}')

        # Log final metrics for the epoch
        self.log("avg_val_loss", avg_loss)
        self.log("avg_val_acc", avg_acc)
        
        

    def test_step(self, batch, batch_idx):
        out = self(batch)

    def accuracy(self, out, y, mask):
        pred = out.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().float()
        return correct / mask.sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=self.factor, patience=self.patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "avg_val_acc"}



def main():
    # Load and log experiment configuration
    config = load_config()
    logger.info(config)

    # Set random seed
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        pl.seed_everything(manual_seed)
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    dataset_config = config["dataset"]
    root_dir = dataset_config["root_dir"]
    train_val_dirs = dataset_config["train_val_dirs"]
    test_dirs = dataset_config["test_dirs"]
    split_mode = dataset_config.get("split_mode", "spatio")
    
    trainer_config = config["trainer"]
    batch_size = trainer_config.get("batch_size", 1)
    model_config = config["model"]
    edge_vector_size = model_config.get("edge_vector_size", 10)
    time_window_size = model_config.get("time_window_size", 60)
    train_dataset = CustomGraphDataset(root_dir, train_val_dirs, test_dirs, split_mode=split_mode, phase="train", max_node_feat_len=time_window_size, max_edge_feat_len=edge_vector_size)
    val_dataset = CustomGraphDataset(root_dir, train_val_dirs, test_dirs, split_mode=split_mode, phase="val", max_node_feat_len=time_window_size, max_edge_feat_len=edge_vector_size)
    DisplayDatasetInfo(train_dataset, "train")
    DisplayDatasetInfo(val_dataset, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    checkpoint_dir = trainer_config.get("checkpoint_dir", "./checkpoints/")
    epochs = trainer_config.get("epochs", 100)
    validate_after_epochs = trainer_config.get("validate_after_epochs", 1)
    early_stop_patience = trainer_config.get("early_stop_patience", 30)

    optimizer_config = config["optimizer"]
    learning_rate = optimizer_config.get("learning_rate", "0.001")
    weight_decay = optimizer_config.get("weight_decay", "0.00001")
    factor = optimizer_config.get("factor", "0.5")
    patience = optimizer_config.get("patience", "10")
    model = get_model(config)
    logger.info(f'The details of the model: {model}')
    lightning_model = GNNLightning(model=model, lr=learning_rate, weight_decay=weight_decay, factor=factor, patience=patience)

    # Early stopping
    early_stopping = EarlyStopping(monitor="avg_val_acc", patience=early_stop_patience, mode="max", verbose=True)
    
    # Ensure the directory is clean before starting a new training session
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)  # Deletes old model files
    os.makedirs(checkpoint_dir, exist_ok=True)  # Creates a fresh directory
    
    # Define ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="avg_val_acc",  # Metric to monitor
        filename="best_model",  # Name of the saved file
        save_top_k=1,           # Save only the best model
        mode="max",             # "min" for loss (lower is better), "max" for accuracy
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,  # Total number of epochs
        check_val_every_n_epoch=validate_after_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[early_stopping, checkpoint_callback],
        deterministic=True,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1
    )

    # Train
    trainer.fit(lightning_model, train_loader, val_loader)





if __name__ == '__main__':
    main()
