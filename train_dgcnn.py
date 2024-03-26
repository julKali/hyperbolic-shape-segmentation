import os
import fire
from pprint import pprint
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Literal
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from einops import rearrange

from x_dgcnn import DGCNN_Seg
from dataset.shapenet import ShapeNetPart
from utils import get_ins_mious, Poly1FocalLoss


class LitModel(pl.LightningModule):
    def __init__(self, n_points, k, dropout, lr, weight_decay, batch_size, epochs, warm_up, optimizer, loss):
        super().__init__()
        self.save_hyperparameters()
        self.warm_up = warm_up
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.net = DGCNN_Seg(k=k, in_dim=3, out_dim=50, n_category=16, dropout=dropout)

        if loss == 'cross_entropy':
            self.criterion = F.cross_entropy
        elif loss == 'poly1_focal':
            self.criterion = Poly1FocalLoss()
        else:
            raise NotImplementedError

        # metrics: instance mIoU & class mIoU
        self.val_inst_mious = []
        self.val_cls = []

    def forward(self, x, xyz, cls):
        return self.net(x, xyz, cls)

    def training_step(self, batch, batch_idx):
        x, cls, y = batch
        x = rearrange(x, 'b n d -> b d n')
        pred = self(x, x[:, :3, :].clone(), cls[:, 0])
        loss = self.criterion(pred, y)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, cls, y = batch
        x = rearrange(x, 'b n d -> b d n')
        pred = self(x, x[:, :3, :].clone(), cls[:, 0])
        loss = self.criterion(pred, y)
        self.log('val_loss', loss, prog_bar=True)

        self.val_inst_mious.append(get_ins_mious(pred.argmax(1), y, cls, ShapeNetPart.cls2parts))
        self.val_cls.append(cls)

    def on_validation_epoch_end(self):
        val_inst_mious = torch.cat(self.val_inst_mious)
        val_cls = torch.cat(self.val_cls)[:, 0]
        cls_mious = []
        for cls in range(len(ShapeNetPart.cls2parts)):
            if (val_cls == cls).sum() > 0:
                cls_mious.append(val_inst_mious[val_cls == cls].mean())
        self.log('val_inst_miou', torch.cat(self.val_inst_mious).mean(), prog_bar=True)
        self.log('val_cls_miou', torch.stack(cls_mious).mean(), prog_bar=True)
        self.val_inst_mious.clear()
        self.val_cls.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=self.trainer.estimated_stepping_batches, max_lr=self.lr,
            pct_start=self.warm_up / self.trainer.max_epochs, div_factor=10, final_div_factor=100)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        H = self.hparams
        return DataLoader(ShapeNetPart(n_points=H.n_points, partition='trainval'), batch_size=H.batch_size,
                          num_workers=4, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        H = self.hparams
        return DataLoader(ShapeNetPart(n_points=H.n_points, partition='test'), batch_size=H.batch_size, num_workers=4,
                          shuffle=False, pin_memory=True)


def run(k=40,
        n_points=2048,
        dropout=0.3,
        batch_size=32,
        lr=1e-3,
        epochs=100,
        warm_up=10,
        loss: Literal['cross_entropy', 'poly1_focal'] = 'cross_entropy',
        optimizer='adamw',
        weight_decay=1e-2,
        gradient_clip_val=0,
        version='dgcnn',
        offline=False):
    # print all hyperparameters
    pprint(locals())
    pl.seed_everything(42)

    os.makedirs('wandb', exist_ok=True)
    logger = WandbLogger(project='shapenet_part_experiments', name=version, save_dir='wandb', offline=offline)
    model = LitModel(n_points=n_points, k=k, dropout=dropout, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                     epochs=epochs, warm_up=warm_up, optimizer=optimizer, loss=loss)
    callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(logger=logger, accelerator='cuda', max_epochs=epochs, callbacks=[callback],
                         gradient_clip_val=gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(run)
