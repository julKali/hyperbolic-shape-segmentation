import os
import fire
from pprint import pprint
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from einops import rearrange
from typing import Literal

from pointnet import PointNet2PartSegMSG, PointNet2PartSegSSG, enable_taichi
from dataset.shapenet import ShapeNetPart
from utils import get_ins_mious, Poly1FocalLoss


#enable_taichi()

import geoopt
from hyp_mlr import Distance2PoincareHyperplanes

# level 0: shape, level 1: pixel
def hier_probs(logits):
    # p(y) = p(class(y)) * p(y) = exp(logits[c_y])/exp(logits[c_1,...,c_2]).sum() * ...
    pass

class Net(torch.nn.Module):

    def __init__(self, net, mlr):
        super().__init__()
        self.net = net
        self.mlr = mlr

    def forward(self, x, xyz, cls):
        l0 = self.net(x, xyz, cls) # ([32, 50, 2048])
        l1 = self.mlr(l0.permute(0,2,1)) # 32, 2048, 50
        return l1.permute(0,2,1)


class LitModel(pl.LightningModule):
    def __init__(self, n_points, model, dropout, lr, batch_size, epochs, warm_up, optimizer, loss):
        super().__init__()
        self.save_hyperparameters()
        self.warm_up = warm_up
        self.lr = lr
        self.batch_size = batch_size

        # todo: here we can change 50 to 3 or something ...

        if model == "ssg":
            self.net = PointNet2PartSegSSG(3, 5, 16, dropout=dropout)
        elif model == "msg":
            self.net = PointNet2PartSegMSG(3, 5, 16, dropout=dropout)
        else:
            raise NotImplementedError

        self.ball = geoopt.PoincareBall()
        mlr = Distance2PoincareHyperplanes(5, 50, ball=self.ball)
        #eucl = torch.nn.Sequential(torch.nn.Linear(40,50)) # euclidean projection, try with activation?
        self.net = Net(self.net, mlr)

        if loss == 'cross_entropy':
            self.criterion = F.cross_entropy
        elif loss == 'poly1_focal':
            self.criterion = Poly1FocalLoss()

        self.val_inst_mious = []
        self.val_cls = []

    # x = points of shape, xyz = segm per point, cls = class of shape
    def forward(self, x, xyz, cls):
        #print("input", x.shape, xyz.shape, cls.shape)
        out = self.net(x, xyz, cls)
        #print("output", out.shape)
        return out

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
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        else:
            raise NotImplementedError
        optimizer = geoopt.optim.RiemannianAdam(self.net.parameters(), lr=1e-4)
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


def run(n_points=2048,
        model: Literal['ssg', 'msg'] = 'ssg',
        lr=1e-3,
        epochs=100,
        batch_size=32,
        warm_up=10,
        optimizer='adamw',
        loss: Literal['cross_entropy', 'poly1_focal'] = 'cross_entropy',
        dropout=0.5,
        gradient_clip_val=0,
        version='pointnet2_ssg',
        offline=False):
    # print all hyperparameters
    pprint(locals())
    pl.seed_everything(42)

    os.makedirs('wandb', exist_ok=True)
    logger = WandbLogger(project='shapenet_part_experiments', name=version, save_dir='wandb', offline=offline)
    model = LitModel(n_points=n_points, model=model, dropout=dropout, batch_size=batch_size, epochs=epochs, lr=lr,
                     warm_up=warm_up, optimizer=optimizer, loss=loss)
    callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(logger=logger, accelerator='cuda', max_epochs=epochs, callbacks=[callback],
                         gradient_clip_val=gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(run)
