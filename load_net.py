from train_pointnet import LitModel1
from dataset.shapenet import ShapeNetPart
from torch.utils.data import DataLoader
from einops import rearrange

net = LitModel1.load_from_checkpoint("wandb/hyperbolic/eu9pudl4/checkpoints/last.ckpt")

H = net.hparams
n_points = 2048
batch_sz = 32
val_data = DataLoader(ShapeNetPart(n_points=n_points, partition='test'), batch_size=batch_sz, num_workers=4, shuffle=False, pin_memory=True)


#print(net.net.mlr.dirs)

val_data_iter = iter(val_data)

x,cls,y = next(val_data_iter)
x = rearrange(x, 'b n d -> b d n').cuda()
#pred = self(x, x[:, :3, :].clone(), cls[:, 0])
logits = net(x)
probs = net.calc_probs(logits)
mlr = net.net.mlr
p = mlr.points
#print(net.net.mlr.points.data.pow(2).sum(dim=-1), 1/net.net.mlr.ball.c.item())
print(mlr.ball.lambda_x(p))
#print(net.net.mlr.points)
#adj_logits = probs.log()
#print(fst)
