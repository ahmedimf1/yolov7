srun --mem=80gb --gres=gpu:3 --partition=gpu -N 1 -n 10 --time=10:02:00 --pty /bin/bash -c "$HOME/scratch/scripts/jupyter_start.sh;$SHELL"


cd ~/scratch/yolov7
source /users/aem603/scratch/anaconda3/bin/activate yolov7-seg
~/scratch/yolov7/check_cuda_devices.py
tensorboard --bind_all --logdir runs/train &!
PYTHONPATH=$PWD python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 64 --data data/Wheat.yaml --img-size 1024 1024 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name yolov7-wheat --hyp data/hyp.scratch.custom.yaml --epochs 25
# PYTHONPATH=$PWD python train.py --workers 8 --device 0,1,2,3 --batch-size 32 --data data/Wheat.yaml --img-size 1024 1024 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name yolov7-wheat --hyp data/hyp.scratch.custom.yaml --epochs 50 &!
# PYTHONPATH=$PWD python -m torch.distributed.launch --node_rank 0 --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/Wheat.yaml --img-size 1024 1024 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name yolov7-wheat --hyp data/hyp.scratch.custom.yaml --epochs 50 &
# PYTHONPATH=$PWD python -m torch.distributed.run --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr "gpu02" --master_port 9527 train.py --workers 8 --device 0,1,2,3 --batch-size 128 --data data/Wheat.yaml --img-size 1024 1024 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name yolov7-wheat --hyp data/hyp.scratch.custom.yaml --epochs 50 &
# PYTHONPATH=$PWD python -m torch.distributed.run --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr "gpu02" --master_port 9527 train.py --workers 8 --device 0,1,2,3 --batch-size 128 --data data/Wheat.yaml --img-size 1024 1024 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name yolov7-wheat --hyp data/hyp.scratch.custom.yaml --epochs 50 &
PYTHONPATH=$PWD python test.py --device 0,1,2,3 --batch-size 32 --data data/Wheat.yaml  --weights /users/aem603/scratch/yolov7/runs/train/yolov7-wheat11/weights/best.pt --name yolov7-wheat11

watch -d -n 0.5 nvidia-smi

PYTHONPATH=$PWD python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 test.py --device 0,1,2,3 --sync-bn --batch-size 32 --data data/Wheat.yaml  --weights /users/aem603/scratch/yolov7/runs/train/yolov7-wheat11/weights/best.pt --name yolov7-wheat11

PYTHONPATH=$PWD python test.py --device 0,1,2,3 --batch-size 32 --data data/Wheat.yaml  --weights /users/aem603/scratch/yolov7/runs/train/yolov7-wheat16/weights/best.pt --name yolov7-wheat16
