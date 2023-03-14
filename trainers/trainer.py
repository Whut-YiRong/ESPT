import os
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
import random
from tqdm import tqdm
from .eval import meta_test
from torch.backends import cudnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(filename):
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", help="optimizer", choices=['adam', 'sgd'])
    parser.add_argument("--lr", help="initial learning rate", type=float)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs", type=int)
    parser.add_argument("--epoch_size", help="number of episode in each epoch", type=int)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float)

    parser.add_argument("--weight", help="loss weight", type=float)
    parser.add_argument("--alpha", help="parameter for linear regression", type=float)

    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", action="store_true")
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int)
    parser.add_argument('--decay_epoch', nargs='+', help='epochs that cut lr', type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")
    parser.add_argument("--no_val", help="don't use validation set", action="store_true")

    parser.add_argument("--train_way", help="training way", type=int)
    parser.add_argument("--test_way", help="test way", type=int)
    parser.add_argument("--train_support_shot", help="number of support images per class for meta-training", type=int)
    parser.add_argument("--train_query_shot", help="number of query images per class for meta-training", type=int)
    parser.add_argument("--test_support_shot", nargs='+', help="number of support images per class for test", type=int)
    parser.add_argument("--test_query_shot", help="number of query images per class for test", type=int)
    parser.add_argument("--train_transform_type", help="size transformation typ for training", type=int)
    parser.add_argument("--test_transform_type", help="size transformation type for test", type=int)
    parser.add_argument("--val_episode", help="number of episodes for validation", type=int, default=2000)
    parser.add_argument("--test_episode", help="number of episodes for test", type=int, default=10000)
    parser.add_argument("--detailed_name", help="whether include training details in the name", action="store_true")

    args = parser.parse_args()
    return args


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    else:
        raise Exception('unknown optimizer_type')

    if args.decay_epoch is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.gamma)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch, gamma=args.gamma)

    return optimizer, scheduler


class Path_Manager:
    def __init__(self, few_shot_path, args):
        self.train = os.path.join(few_shot_path, 'train')
        if args.pre:
            self.test = os.path.join(few_shot_path, 'test_pre')
            self.val = os.path.join(few_shot_path, 'val_pre') if not args.no_val else self.test
        else:
            self.test = os.path.join(few_shot_path, 'test')
            self.val = os.path.join(few_shot_path, 'val') if not args.no_val else self.test


class Train_Manager:
    def __init__(self, args, path_manager, epoch_trainer):
        self.args = args
        setting = '-seed' + str(self.args.seed) + '-alpha' + str(self.args.alpha) + '-weight' + str(self.args.weight) +\
                  '-lr' + str(self.args.lr) + '-way' + str(self.args.train_way) + '-decay_epoch' + str(self.args.decay_epoch)
        name = 'ResNet-12' if self.args.resnet else 'Conv-4'
        name = name + setting

        self.logger = get_logger('%s.log' % name)
        self.save_path = 'model_%s_best.pth' % name
        self.save_last_path = 'model_%s_last.pth' % name
        self.epoch_trainer = epoch_trainer
        self.path_manager = path_manager
        self.display_args()

    def display_args(self):
        self.logger.info('display all the hyper-parameters in args:')
        for arg in vars(self.args):
            value = getattr(self.args, arg)
            if value is not None:
                self.logger.info('%s: %s' % (str(arg), str(value)))
        self.logger.info('------------------------')

    def train(self, model):
        optimizer, scheduler = get_optimizer(model, self.args)
        val_way = self.args.test_way
        val_shot = self.args.train_support_shot
        self.logger.info("start training!")

        iter_counter = 0
        best_val_acc = 0
        best_epoch = 0

        model.train()
        model.cuda()

        for e in tqdm(range(self.args.epoch)):
            print(model.scale)
            
            if model.is_pretraining:
                print(model.alpha_pretrain)
            else:
                print(model.alpha)
            model.train()
            
            # training for each epoch
            iter_counter, train_acc, avg_cls_loss, avg_ssl_loss = self.epoch_trainer(
                model=model, optimizer=optimizer, iter_counter=iter_counter, weight=self.args.weight
            )
            self.logger.info('avg_cls_loss: %.5f\t avg_ssl_loss: %.5f\t' % (avg_cls_loss, avg_ssl_loss))

            if (e + 1) % self.args.val_epoch == 0:
                self.logger.info("")
                self.logger.info("epoch %d/%d, iter %d:" % (e + 1, self.args.epoch, iter_counter))
                self.logger.info("train_acc: %.3f" % train_acc)

                with torch.no_grad():
                    val_acc, val_interval = meta_test(
                        data_path=self.path_manager.val, model=model, pre=self.args.pre,
                        transform_type=self.args.test_transform_type, episode=self.args.val_episode,
                        way=val_way, support_shot=val_shot, query_shot=self.args.test_query_shot, seed=1
                    )

                self.logger.info('val_%d-way-%d-shot_acc: %.3f\t%.3f' % (val_way, val_shot, val_acc, val_interval))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = e + 1
                    if not self.args.no_val:
                        torch.save(model.state_dict(), self.save_path)
                    self.logger.info('BEST!')

            scheduler.step()

        self.logger.info('training finished!')
        torch.save(model.state_dict(), self.save_last_path)

        self.logger.info('------------------------')
        self.logger.info('the best epoch is %d/%d' % (best_epoch, self.args.epoch))
        self.logger.info('the best %d-way %d-shot val acc is %.3f' % (val_way, val_shot, best_val_acc))

    def evaluate(self, model, seed=0):
        self.logger.info('------------------------')
        self.logger.info('evaluating on test set:')

        with torch.no_grad():
            model.load_state_dict(torch.load(self.save_path))
            for shot in self.args.test_support_shot:
                mean, interval = meta_test(
                    data_path=self.path_manager.test, model=model, pre=self.args.pre,
                    transform_type=self.args.test_transform_type, episode=self.args.test_episode,
                    way=self.args.test_way, support_shot=shot, query_shot=self.args.test_query_shot, seed=seed
                )
                self.logger.info('Best: %d-way-%d-shot acc: %.2f\t%.2f' % (self.args.test_way, shot, mean, interval))