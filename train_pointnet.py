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

from pointnet import PointNetSeg
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

    def forward(self, x):
        l0 = self.net(x) # ([32, 50, 2048])
        l1 = self.mlr(l0.permute(0,2,1)) # 32, 2048, 50
        return l1.permute(0,2,1)


class LitModel1(pl.LightningModule):
    def __init__(self, n_points, hier, dim, model, dropout, lr, batch_size, epochs, warm_up, optimizer, loss):
        super().__init__()
        self.save_hyperparameters()
        self.warm_up = warm_up
        self.lr = lr
        self.batch_size = batch_size

        self.hier = hier
        class_dim = 50 if not hier else 16 + 50

        self.net = PointNetSeg(3, dim)

        self.ball = geoopt.PoincareBall()
        if model == "hyp":
            last = Distance2PoincareHyperplanes(dim, class_dim, ball=self.ball)
        elif model == "eucl":
            last = torch.nn.Sequential(torch.nn.Linear(dim, class_dim)) # euclidean projection, try with activation?
        else:
            raise NotImplementedError()
        self.net = Net(self.net, last)

        if loss == 'cross_entropy':
            self.criterion = F.cross_entropy
        elif loss == 'poly1_focal':
            self.criterion = Poly1FocalLoss()

        self.val_inst_mious = []
        self.val_cls = []

    # x = points of shape, xyz = segm per point, cls = class of shape
    def forward(self, x):#, xyz, cls):
        #print("input", x.shape, xyz.shape, cls.shape)
        out = self.net(x)
        #print("output", out.shape)
        return out

    # logits: B, C, PT
    def adjust_logits(self, logits):
        raise NotImplementedError()
        if self.hier:
            B,C,PT = logits.shape
            if logits.isnan().any():
                ValueError("logits nan")

            probs = torch.zeros((B, 50, PT)).to(logits.device)

            #logits_min = logits.min()
            #if logits_min < -100:
            #    print("Clamping logits to >= -100, before: >=", logits_min)
            #logits = logits.clamp(min=-100) # so that exp() will not return 0

            # class level:
            denom = logits[:,:16,:].exp().sum(dim=1).clamp(min=1e-8) # B,PT
            for cls_i in range(16):
                ps_i = torch.tensor(ShapeNetPart.cls2parts[cls_i]).to(logits.device) # indices of parts of class cls_i, |P|
                probs[:,ps_i,:] = logits[:, cls_i,:].exp().unsqueeze(1) / denom.unsqueeze(1) # B,|P|,PT
                part_probs = logits[:,16+ps_i,:].exp() # logits of parts of that class, B,|P|,PT
                denom2 = part_probs.sum(dim=1).clamp(min=1e-8) # B,PT
                probs[:,ps_i,:] *= part_probs / denom2.unsqueeze(1)

            # sanity check
            if (probs < 0).any():
                print(logits)
                print(probs)
                raise ValueError("probs negative")
            elif probs.isnan().any():
                print(logits)
                print("---------")
                print(probs)
                raise ValueError("probs nan")

            probs = probs.clamp(min=1e-8)
            return probs.log()
        else:
            return logits

    # logits: B, C, PT
    def adjust_logits2(self, logits):
        if self.hier:
            B,C,PT = logits.shape
            if logits.isnan().any():
                print("logits nan")

            logits = logits - logits.max(dim=1, keepdim=True)[0] # B,C,PT - B,C,PT => broadcast to B,C,PT - B,C,PT

            adj_logits = torch.zeros((B, 50, PT)).to(logits.device)

            # class level:
            denom = logits[:,:16,:].exp().sum(dim=1).clamp(min=1e-15).log() # B,PT
            for cls_i in range(16):
                ps_i = torch.tensor(ShapeNetPart.cls2parts[cls_i]).to(logits.device) # indices of parts of class cls_i, |P|
                adj_logits[:,ps_i,:] = logits[:, cls_i,:].unsqueeze(1) - denom.unsqueeze(1) # B,|P|,PT
                part_logits = logits[:,16+ps_i,:] # logits of parts of that class, B,|P|,PT
                denom2 = part_logits.exp().sum(dim=1).clamp(min=1e-15).log() # B,PT
                adj_logits[:,ps_i,:] += part_logits - denom2.unsqueeze(1)

            # sanity check
            if adj_logits.isnan().any():
                print(logits)
                print("---------")
                print(adj_logits)
                raise Error("adj_logits nan")

            return adj_logits
        else:
            return logits

    def training_step(self, batch, batch_idx):
        #self.log("points_val_nan", self.net.mlr.points.data.isnan().sum())
        #self.log("dirs_val_nan", self.net.mlr.dirs.data.isnan().sum())
        x, cls, y = batch
        # cls: B, 1
        # y: B, PT
        x = rearrange(x, 'b n d -> b d n')
        #pred = self(x, x[:, :3, :].clone(), cls[:, 0])
        logits = self(x)
        adj_logits = self.adjust_logits2(logits)
        #self.log("min_prob", probs.min(), prog_bar=True)
        loss = F.cross_entropy(adj_logits, y)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, cls, y = batch
        x = rearrange(x, 'b n d -> b d n')
        #pred = self(x, x[:, :3, :].clone(), cls[:, 0])
        logits = self(x)
        adj_logits = self.adjust_logits2(logits)
        loss = F.cross_entropy(adj_logits, y)
        self.log('val_loss', loss, prog_bar=True)

        self.val_inst_mious.append(get_ins_mious(adj_logits.argmax(1), y, cls, ShapeNetPart.cls2parts))
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
            pass
        optimizer = geoopt.optim.RiemannianAdam(self.net.parameters(), lr=self.lr)
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
        model: Literal['hyp', 'eucl'] = 'hyp',
        dim= 50,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        warm_up=10,
        optimizer='radamw',
        loss: Literal['cross_entropy', 'poly1_focal'] = 'cross_entropy',
        dropout=0.5,
        gradient_clip_val=0,
        version='pointnet',
        hier=True,
        offline=False,
        load_ckpt=None,
        save_ckpt_n=10,
        ):
    # print all hyperparameters
    pprint(locals())
    pl.seed_everything(42)

    #torch.autograd.set_detect_anomaly(True)

    os.makedirs('wandb', exist_ok=True)
    logger = WandbLogger(project='hyperbolic', name=version, save_dir='wandb', offline=offline)
    if load_ckpt is None:
        model = LitModel1(n_points=n_points, dim=dim, hier=hier, model=model, dropout=dropout, batch_size=batch_size, epochs=epochs, lr=lr,
                        warm_up=warm_up, optimizer=optimizer, loss=loss)
    else:
        model = LitModel1.load_from_checkpoint(load_ckpt)
    callback = ModelCheckpoint(save_last=True, every_n_epochs=save_ckpt_n)

    logger.watch(model, log='all')

    trainer = pl.Trainer(logger=logger, accelerator='cuda', max_epochs=epochs, callbacks=[callback],
                         gradient_clip_val=gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(run)
