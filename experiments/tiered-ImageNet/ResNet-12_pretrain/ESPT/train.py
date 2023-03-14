import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, meta_train
from datasets import dataloaders
from models.ESPT_Euclidean import ESPT

args = trainer.train_parser()
torch.cuda.set_device(args.gpu)
trainer.set_seed(args.seed)

with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
few_shot_path = os.path.join(data_path, 'tiered-ImageNet_DeepEMD')
path_manager = trainer.Path_Manager(few_shot_path=few_shot_path, args=args)

train_loader = dataloaders.normal_dataloader(
    data_path=path_manager.train, transform_type=args.train_transform_type, batch_size=args.batch_size
)

num_cat = len(train_loader.dataset.classes)

model = ESPT(
    way=args.train_way, support_shot=args.train_support_shot, query_shot=args.train_query_shot, alpha=args.alpha,
    resnet=args.resnet, seed=args.seed, is_pretraining=True, num_cat=num_cat
)

epoch_trainer = partial(meta_train.pretrain, train_loader=train_loader)
train_manager = trainer.Train_Manager(args, path_manager=path_manager, epoch_trainer=epoch_trainer)

train_manager.train(model)
train_manager.evaluate(model, seed=0)
