import multiprocessing
import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnext50_32x4d


class ResNeXtLightning(L.LightningModule):
    def __init__(
        self,
        n_classes=10,
        learning_rate=0.01,
        batch_size=128,
        num_workers=8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnext50_32x4d()
        # Modify first conv layer for CIFAR10's 32x32 images (instead of ImageNet's 224x224)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()  # Remove maxpool as we have smaller images
        self.model.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics to TensorBoard
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics to TensorBoard
        self.log("val/loss", loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics to TensorBoard
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = CIFAR10(root="data", train=True, transform=transform, download=True)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = CIFAR10(root="data", train=False, transform=transform, download=True)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )


def main():
    rank = int(
        os.getenv("SLURM_PROCID")
    )  # Get individual process ID from SLURM environment variable.

    n_gpus = torch.cuda.device_count()
    n_cpus = multiprocessing.cpu_count()
    n_workers = min(4 * n_gpus, n_cpus // n_gpus)
    if rank == 0:
        print("Number of GPUs:", n_gpus)
        print("Number of CPUs:", n_cpus)
        print(f"Number of workers for DataLoader: {n_workers}")

    model = ResNeXtLightning(
        num_workers=4 * n_gpus,
        learning_rate=0.01,
        batch_size=128,
    )
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="16-mixed",
        logger=L.pytorch.loggers.TensorBoardLogger(
            "lightning_logs", name="resnext_cifar10"
        ),
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
