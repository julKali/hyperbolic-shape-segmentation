from train_pointnet2 import LitModel
from dataset.shapenet import ShapeNetPart
from torch.utils.data import DataLoader

net = LitModel.load_from_checkpoint("wandb/shapenet_part_experiments/5nv35crq/checkpoints/last.ckpt")

H = net.hparams
n_points = 1024
batch_sz = 32
val_data = DataLoader(ShapeNetPart(n_points=n_points, partition='test'), batch_size=batch_sz, num_workers=4, shuffle=False, pin_memory=True)

print(net.net.mlr.points.data.pow(2).sum(dim=-1).sqrt())
#print(net.net.mlr.dirs)

#val_data_iter = iter(val_data)

#fst = next(iter(val_data))
#print(fst)
