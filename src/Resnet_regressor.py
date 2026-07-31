import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models

class ResNetRegressorPLM(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, backbone="resnet18"):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        if backbone == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        elif backbone == "resnet34":
            self.model = models.resnet34(weights="IMAGENET1K_V1")
        else:
            raise ValueError("Unsupported backbone")

        # Replace final FC layer with regression head (2 outputs: lat, lon)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.compute_loss(outputs, targets)
        rmse = torch.sqrt(loss)
        self.log('train_loss', loss)
        self.log('train_rmse', rmse)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.compute_loss(outputs, targets)
        rmse = torch.sqrt(loss)
        self.log('val_loss', loss)
        self.log('val_rmse', rmse)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)