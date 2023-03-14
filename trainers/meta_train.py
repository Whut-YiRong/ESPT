import torch
import torch.nn as nn
import torch.nn.functional as F


# def get_auxiliary_loss1(support):
#     way = support.size(0)
#     shot = support.size(1)
#     print(support.norm(2).unsqueeze(-1))
#     support = support / support.norm(2).unsqueeze(-1)
#     L1 = torch.zeros((way**2-way)//2).long().cuda()
#     L2 = torch.zeros((way**2-way)//2).long().cuda()
#     counter = 0
#     for i in range(way):
#         for j in range(i):
#             L1[counter] = i
#             L2[counter] = j
#             counter += 1
#     s1 = support.index_select(0, L1) # (s^2-s)/2, s, d
#     s2 = support.index_select(0, L2) # (s^2-s)/2, s, d
#     dists = s1.matmul(s2.permute(0,2,1)) # (s^2-s)/2, s, s
#     assert dists.size(-1)==shot
#     frobs = dists.pow(2).sum(-1).sum(-1)
#     print( support.norm(2).unsqueeze(-1))
#     return frobs.sum().mul(0.03)


def meta_train_Proto(train_loader, model, optimizer, iter_counter):
    model.train()
    way = model.way
    query_shot = model.query_shot
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()

    avg_loss = 0
    avg_acc = 0
    epoch_size = 0
    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        epoch_size += 1
        inp = inp.cuda()
        log_prediction = model(inp)

        loss = criterion(log_prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss.item()

    avg_acc = avg_acc / epoch_size
    avg_loss = avg_loss / epoch_size

    return iter_counter, avg_acc, avg_loss, avg_loss, avg_loss


def pretrain(train_loader, model, optimizer, iter_counter, weight=None):
    model.train()
    criterion = nn.NLLLoss().cuda()

    avg_cls_loss = 0
    avg_ssl_loss = 0
    avg_acc = 0
    epoch_size = 0

    for i, (inp, target) in enumerate(train_loader):
        iter_counter += 1
        epoch_size += 1
        inp = inp.cuda()
        target = target.cuda()
        batch_size = target.size(0)

        log_prediction = model.pretrain_forward(inp=inp)
        loss = criterion(log_prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / batch_size

        avg_acc += acc
        avg_cls_loss += loss.item()
        avg_ssl_loss += loss.item()

    avg_acc = avg_acc / epoch_size
    avg_cls_loss = avg_cls_loss / epoch_size
    avg_ssl_loss = avg_ssl_loss / epoch_size

    return iter_counter, avg_acc, avg_cls_loss, avg_ssl_loss


def meta_train_ESPT(train_loader, model, optimizer, iter_counter, weight):
    model.train()
    way = model.way
    query_shot = model.query_shot
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()

    transform_list = [1, 2, 3]
    transform_type = 'rotation'

    print(weight)
    print(transform_list)

    avg_cls_loss = 0
    avg_ssl_loss = 0
    avg_acc = 0
    epoch_size = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        epoch_size += 1
        inp = inp.cuda()
        log_prediction_ori, weight_similarity = model.ESPT_forward(
            inp=inp, transform_list=transform_list, transform_type=transform_type
        )

        cls_loss = criterion(log_prediction_ori, target)
        ssl_loss = weight * weight_similarity
        loss = cls_loss + ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction_ori, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_cls_loss += cls_loss.item()
        avg_ssl_loss += ssl_loss.item()

    avg_acc = avg_acc / epoch_size
    avg_cls_loss = avg_cls_loss / epoch_size
    avg_ssl_loss = avg_ssl_loss / epoch_size

    return iter_counter, avg_acc, avg_cls_loss, avg_ssl_loss


def meta_train_LR(train_loader, model, optimizer, iter_counter, weight=None):
    model.train()
    way = model.way
    query_shot = model.query_shot
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()

    avg_cls_loss = 0
    avg_ssl_loss = 0
    avg_acc = 0
    epoch_size = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        epoch_size += 1
        inp = inp.cuda()
        log_prediction = model.LR_forward(inp=inp)

        loss = criterion(log_prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_cls_loss += loss.item()
        avg_ssl_loss += loss.item()

    avg_acc = avg_acc / epoch_size
    avg_cls_loss = avg_cls_loss / epoch_size
    avg_ssl_loss = avg_ssl_loss / epoch_size

    return iter_counter, avg_acc, avg_cls_loss, avg_ssl_loss
