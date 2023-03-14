import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.append('..')
from datasets import dataloaders


def get_score(acc_list):
    mean = np.mean(acc_list)
    interval = 1.96 * np.sqrt(np.var(acc_list) / len(acc_list))

    return mean, interval


def meta_test(data_path, model, pre, transform_type, episode, way, support_shot, query_shot, seed, return_list=False):
    model.eval()
    target = torch.LongTensor([i//query_shot for i in range(query_shot * way)]).cuda()
    eval_loader = dataloaders.meta_dataloader(
        data_path=data_path, is_training=False, pre=pre, transform_type=transform_type, epoch_size=episode,
        way=way, support_shot=support_shot, query_shot=query_shot, seed=seed
    )

    acc_list = []
    for i, (inp, _) in tqdm(enumerate(eval_loader)):
        inp = inp.cuda()
        max_index = model.meta_test(inp, way=way, support_shot=support_shot, query_shot=query_shot)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way
        acc_list.append(acc)

    if return_list:
        return np.array(acc_list)
    else:
        mean, interval = get_score(acc_list)
        return mean, interval
