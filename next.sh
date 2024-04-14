#python train_pointnet.py --model=eucl --dim=2 --n_points=1024 --batch_size=32 --hier=False --version="eucl;dim=2;\!hier;cls_val"
#python train_pointnet.py --model=hyp --dim=2 --n_points=1024 --batch_size=32 --hier=True --version="hyp;dim=2;hier;cls_val"
#python train_pointnet.py --model=eucl --dim=2 --n_points=1024 --batch_size=32 --hier=True --version="eucl;dim=2;hier;cls_val"
#python train_pointnet.py --model=hyp --dim=3 --n_points=1024 --batch_size=32 --hier=False --version="hyp;dim=3;\!hier;cls_val"
#python train_pointnet.py --model=eucl --dim=3 --n_points=1024 --batch_size=32 --hier=False --version="eucl;dim=3;\!hier;cls_val"
#python train_pointnet.py --model=hyp --dim=50 --n_points=2048 --batch_size=16 --hier=False --version="hyp;dim=50;\!hier;cls_val"

#python train_pointnet.py --model=eucl --dim=2 --n_points=2048 --batch_size=16 --hier=True --version="eucl;dim=2;hier"
#python train_pointnet.py --model=hyp --dim=2 --n_points=2048 --batch_size=16 --hier=True --version="hyp;dim=2;hier"

#python train_pointnet.py --model=eucl --dim=3 --n_points=1024 --batch_size=32 --hier=True --version="eucl;dim=3;hier"
#python train_pointnet.py --model=hyp --dim=3 --n_points=1024 --batch_size=32 --hier=True --version="hyp;dim=3;hier"

#python train_pointnet.py --model=hyp --dim=50 --n_points=1024 --batch_size=32 --hier=True --version="hyp;dim=50;hier"
#python train_pointnet.py --model=eucl --dim=50 --n_points=1024 --batch_size=32 --hier=True --version="eucl;dim=50;hier"

#python train_pointnet.py --model=eucl --dim=50 --n_points=2048 --batch_size=16 --hier=False --version="eucl;dim=50;\!hier;cls_val"

python train_pointnet.py --model=hyp --dim=50 --n_points=1024 --batch_size=32 --hier=False --version="hyp;dim=50;\!hier;1024p"
python train_pointnet.py --model=eucl --dim=50 --n_points=1024 --batch_size=32 --hier=False --version="eucl;dim=50;\!hier;1024p"
