import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, meta_train
from datasets import dataloaders
from models.ESPT_Euclidean import ESPT
from utils import util

args = trainer.train_parser()
torch.cuda.set_device(args.gpu)
trainer.set_seed(args.seed)

with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
few_shot_path = os.path.join(data_path, 'tiered-ImageNet_DeepEMD')
path_manager = trainer.Path_Manager(few_shot_path=few_shot_path, args=args)

train_loader = dataloaders.meta_dataloader(
    data_path=path_manager.train, is_training=True, pre=None,
    transform_type=args.train_transform_type, epoch_size=args.epoch_size,
    way=args.train_way, support_shot=args.train_support_shot, query_shot=args.train_query_shot, seed=None
)

model = ESPT(
    way=args.train_way, support_shot=args.train_support_shot, query_shot=args.train_query_shot, alpha=args.alpha,
    resnet=args.resnet, seed=args.seed
)

pretrained_model_path = '../../ResNet-12_pretrain/ESPT/model_ResNet-12-seed1-alpha1.0-weight0.0-lr0.1-way5-decay_epoch[30, 60]_best.pth'
# pretrained_model_path = '../../ResNet-12_pretrain/FRN/model_ResNet-12-miniImageNet.pth'
model.load_state_dict(torch.load(pretrained_model_path, map_location=util.get_device_map(args.gpu)), strict=False)
model.scale = torch.nn.Parameter(torch.FloatTensor([15.0]), requires_grad=True)

epoch_trainer = partial(meta_train.meta_train_ESPT, train_loader=train_loader, transform_list=[1, 2])
train_manager = trainer.Train_Manager(args, path_manager=path_manager, epoch_trainer=epoch_trainer)

train_manager.train(model)
train_manager.evaluate(model, seed=0)
train_manager.evaluate(model, seed=1)
train_manager.evaluate(model, seed=2)
# train_manager.evaluate(model, seed=3)
# train_manager.evaluate(model, seed=4)
# train_manager.evaluate(model, seed=5)
# train_manager.evaluate(model, seed=6)
# train_manager.evaluate(model, seed=7)
# train_manager.evaluate(model, seed=8)
# train_manager.evaluate(model, seed=9)
