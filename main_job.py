import pdb
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizer
import os

from model import ServeNet
from utils import load_data_test, load_data_train, eval_top1, eval_top5
import util

import random

from warnings import simplefilter

# create working servenet model
# change the dataloader to index dataloader
# similarity matrix from gradients
# get subset from the similarity matrix, I mean it would be same as that of craig on CIFAR100.

# from hyper_param import *

ip_file_dir = "./data/"
CLASS_NUM = category_num = 200
max_len = 110
BATCH_SIZE = 256
LEARNING_RATE = 0.01
epochs = 50
def get_train_length():
    train_file = f"{ip_file_dir}{category_num}/train.csv"
    df = pd.read_csv(train_file)
    return len(df.index)
N = get_train_length()
SUBSET_SIZE = 0.1
WORKERS = 1

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
np.seterr(all='ignore')

parser = argparse.ArgumentParser(description='Servenet Coreset')
parser.add_argument('-b', '--subset-size', dest="subset_size", default=0.1, type=float)
parser.add_argument('-c', '--class-num', dest="class_num", default=200, type=int)
parser.add_argument('-r', '--run-num', dest="run_num", default=0, type=int)
parser.add_argument('-bs', '--batch-size', dest="batch_size", default=256, type=int)
parser.add_argument('-e', '--epoch', dest="epochs", default=50, type=int)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def predictions(loader, model):
    """
    Get predictions
    """
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    preds = torch.zeros(N, CLASS_NUM).cuda()
    labels = torch.zeros(N, dtype=torch.int).cuda()
    # end = time.time()
    with torch.no_grad():

        # pdb.set_trace()

        for i, (data, idx) in enumerate(loader):

            descriptions = {'input_ids': data[0].cuda(),
                            'token_type_ids': data[1].cuda(),
                            'attention_mask': data[2].cuda()
                            }

            names = {'input_ids': data[4].cuda(),
                     'token_type_ids': data[5].cuda(),
                     'attention_mask': data[6].cuda()
                     }

            target = data[3].cuda()

            # idx = data[7]

            output = model(names, descriptions)

            preds[idx, :] = nn.Softmax(dim=1)(output)
            labels[idx] = target.int()

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # if i % args.print_freq == 0:
            #     print('Predict: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
            #           .format(i, len(loader), batch_time=batch_time))

    return preds.cpu().data.numpy(), labels.cpu().data.numpy()


class IndexedDataset(Dataset):
    def __init__(self):
        self.data_set = load_data_train(ip_file_dir, category_num, max_len)

    def __getitem__(self, index):
        data = self.data_set[index]
        # Your transformations here (or set it in CIFAR10)
        return data, index

    def __len__(self):
        return len(self.data_set)


def get_gradients(val_loader, model, criterion):
    model.eval()

    preds = torch.zeros(N, CLASS_NUM).cuda()
    labels = torch.zeros(N, dtype=torch.int).cuda()
    loss_list = torch.zeros(N).cuda()

    with torch.no_grad():
        for i, (data, idx) in enumerate(val_loader):
            descriptions = {'input_ids': data[0].cuda(),
                            'token_type_ids': data[1].cuda(),
                            'attention_mask': data[2].cuda()
                            }

            names = {'input_ids': data[4].cuda(),
                     'token_type_ids': data[5].cuda(),
                     'attention_mask': data[6].cuda()
                     }

            target = data[3].cuda()

            # compute output
            output = model(names, descriptions)
            loss = criterion(output, target)
            # pdb.set_trace()
            preds[idx, :] = nn.Softmax(dim=1)(output)
            labels[idx] = target.int()
            loss_list[idx] = loss

    return preds.cpu().data.numpy(), labels.cpu().data.numpy(), loss_list.cpu().data.numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    CLASS_NUM = category_num = args.class_num
    SUBSET_SIZE = args.subset_size
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    N = get_train_length()

    train_data = load_data_train(ip_file_dir, category_num, max_len)
    test_data = load_data_test(ip_file_dir, category_num, max_len)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    indexed_dataset = IndexedDataset()
    indexed_loader = DataLoader(
        indexed_dataset,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True)

    # initialize a random subset
    B = int(SUBSET_SIZE * N)

    # selected_ndx = np.zeros((epochs, B))
    selected_ndx = []
    # selected_wgt = np.zeros((epochs, B))
    selected_wgt = []
    times_selected = np.zeros((len(indexed_loader.dataset)))
    not_selected = np.zeros(epochs)

    order = np.arange(0, N)
    np.random.shuffle(order)
    order = order[:B]
    weight = None
    subset = np.array([x for x in range(N)])
    subset_weight = np.ones(N)

    model = ServeNet(768, CLASS_NUM)

    model.weight_sum.w1 = torch.nn.Parameter(torch.tensor([0.5]))
    model.weight_sum.w2 = torch.nn.Parameter(torch.tensor([0.5]))

    # model.bert_description.requires_grad_(False)
    # model.bert_name.requires_grad_(False)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()

    # pdb.set_trace()
    best_accuracy = 0

    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params_all = sum(p.numel() for p in model.parameters())
    print("Trainable: ", pytorch_total_params_trainable)
    print("All: ", pytorch_total_params_all)

    criterion = torch.nn.CrossEntropyLoss()
    train_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    w11 = model.module.weight_sum.w1
    w22 = model.module.weight_sum.w2
    print("w11: ", w11)
    print("w22: ", w22)

    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    test_acc5_list = []
    train_acc_list = []

    gradient_list_rel = []
    gradient_list_rel_full = []
    gradient_list_norm_all = []
    gradient_list_norm_full = []
    gradient_list_norm_sub = []

    loss_list_rel = []
    loss_list_all = []
    loss_list_sub = []

    gradient_storage = []


    for epoch in range(epochs):
        print("Epoch:{},lr:{}".format(str(epoch + 1), str(optimizer.state_dict()['param_groups'][0]['lr'])))
        # scheduler.step()

        # get gradients:

        if SUBSET_SIZE >= 1:
            train_loader = indexed_loader
        else:
            preds, labels = predictions(indexed_loader, model)
            preds -= np.eye(CLASS_NUM)[labels]

            subset, subset_weight, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                B, preds, 'euclidean', smtk=0, no=0, y=labels, stoch_greedy=0,
                equal_num=False)

            weights = np.zeros(len(indexed_loader.dataset))
            scaled_weight = subset_weight / np.sum(subset_weight) * len(subset_weight)

            # pdb.set_trace()

            # selected_ndx[epoch], selected_wgt[epoch] = subset, scaled_weight
            selected_ndx.append(subset)
            selected_wgt.append(scaled_weight)


            weights[subset] = scaled_weight
            weight = torch.from_numpy(weights).float().cuda()

            times_selected[subset] += 1
            print(f'{np.sum(times_selected == 0) / len(times_selected) * 100:.3f} % not selected yet')
            not_selected[epoch] = np.sum(times_selected == 0) / len(times_selected) * 100
            indexed_subset = torch.utils.data.Subset(indexed_loader, indices=subset)
            train_loader = DataLoader(
                indexed_dataset,
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=WORKERS,
                pin_memory=True)

        model.train()

        # pdb.set_trace()

        if weight is None:
            weight = torch.ones(N).cuda()

        top1 = AverageMeter()
        losses = AverageMeter()

        for i, (data, idx) in enumerate(train_loader):
        # for data, idx in tqdm(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            descriptions = {'input_ids': data[0].cuda(),
                            'token_type_ids': data[1].cuda(),
                            'attention_mask': data[2].cuda()
                            }

            names = {'input_ids': data[4].cuda(),
                     'token_type_ids': data[5].cuda(),
                     'attention_mask': data[6].cuda()
                     }
            label = data[3].cuda()

            outputs = model(names, descriptions)

            loss = train_criterion(outputs, label)
            loss = (loss * weight[idx.long()]).mean()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(outputs.data, label)[0]
            losses.update(loss.item(), data[3].size(0))
            top1.update(prec1.item(), data[3].size(0))

        # pdb.set_trace()
        top_1_acc = eval_top1(model, test_dataloader, category_num)
        top_5_acc = eval_top5(model, test_dataloader)
        epoch_list.append(epoch + 1)
        test_acc_list.append(top_1_acc)
        test_acc5_list.append(top_5_acc)
        train_acc_list.append(top1.avg)
        train_loss_list.append(losses.avg)

        gp, gt, gl = get_gradients(indexed_loader, model, train_criterion)

        g_full_path = f"./files/subset_100b_{CLASS_NUM}c.npz"
        if os.path.isfile(g_full_path):
            g_full = np.load(g_full_path)
        else:
            g_full = None
        if g_full:
            last_epoch_g_full = g_full["all_gradient"][epoch] if g_full else None
        else:
            last_epoch_g_full = None

        first_gradient_all = gp - np.eye(CLASS_NUM)[gt]
        first_gradient_ss = first_gradient_all[subset]

        gradient_storage.append(first_gradient_all.sum(axis=0))

        first_gradient_ss_wt = first_gradient_ss * np.tile(subset_weight, (CLASS_NUM, 1)).T

        first_gradient_error_wt = first_gradient_all.sum(axis=0) - first_gradient_ss_wt.sum(axis=0)
        first_gradient_error_wt_full = last_epoch_g_full - first_gradient_ss_wt.sum(axis=0) if last_epoch_g_full else None

        first_gradient_norm_wt_full = np.linalg.norm(first_gradient_error_wt_full) if first_gradient_error_wt_full else 0
        first_gradient_norm_wt_rel = np.linalg.norm(first_gradient_error_wt) / np.linalg.norm(
            first_gradient_all.sum(axis=0))
        first_gradient_norm_wt_rel_full = first_gradient_norm_wt_full / np.linalg.norm(last_epoch_g_full) if last_epoch_g_full else 0

        first_gradient_norm_all = np.linalg.norm(first_gradient_all.sum(axis=0))
        first_gradient_norm_full = np.linalg.norm(last_epoch_g_full) if last_epoch_g_full else 0
        first_gradient_norm_sub = np.linalg.norm(first_gradient_ss_wt.sum(axis=0))

        loss_all = gl
        loss_ss = gl[subset]
        loss_ss_wt = loss_ss * subset_weight

        loss_error = gl.sum() - loss_ss.sum()
        loss_error_wt = gl.sum() - loss_ss_wt.sum()
        loss_error_wt_rel = loss_error_wt / gl.sum()
        loss_error_all = gl.sum()
        loss_error_sub = loss_ss_wt.sum()

        gradient_list_rel.append(first_gradient_norm_wt_rel)
        gradient_list_rel_full.append(first_gradient_norm_wt_rel_full)
        gradient_list_norm_all.append(first_gradient_norm_all)
        gradient_list_norm_full.append(first_gradient_norm_full)
        gradient_list_norm_sub.append(first_gradient_norm_sub)

        loss_list_rel.append(loss_error_wt_rel)
        loss_list_all.append(loss_error_all)
        loss_list_sub.append(loss_error_sub)

        # if top_1_acc > best_accuracy:
        #     best_accuracy = top_1_acc
        #     torch.save(model, f"./files/snlt3_best3_{CLASS_NUM}_{int(SUBSET_SIZE*100)}b_full")

        print("=======>top1 acc on the test:{}".format(str(top_1_acc)))
        print("=======>top5 acc on the test:{}".format(str(top_5_acc)))
        w11 = model.module.weight_sum.w1
        w22 = model.module.weight_sum.w2
        print("w11: ", w11)
        print("w22: ", w22)

        # pdb.set_trace()

        acc_list = pd.DataFrame(
            {
                'epoch': epoch_list,
                'top1_test_acc': test_acc_list,
                'top5_test_acc': test_acc5_list,
                'train_acc': train_acc_list,
                'train_loss': train_loss_list,

                'rel_grad': gradient_list_rel,
                'rel_grad_full': gradient_list_rel_full,
                'all_grad': gradient_list_norm_all,
                'full_grad': gradient_list_norm_full,
                'sub_grad': gradient_list_norm_sub,

                'rel_loss': loss_list_rel,
                'all_loss': loss_list_all,
                'sub_loss': loss_list_sub,
            }
        )

        acc_list.to_csv(f'./files/sn_{int(SUBSET_SIZE*100)}b_{CLASS_NUM}c.csv')

        if SUBSET_SIZE < 1:
            np.savez(f"./files/subset_{int(SUBSET_SIZE*100)}b_{CLASS_NUM}c",
                     subset=selected_ndx, weight=selected_wgt, all_gradient=gradient_storage)
        else:
            np.savez(f"./files/subset_{int(SUBSET_SIZE * 100)}b_{CLASS_NUM}c", all_gradient=gradient_storage)
    # print("=======>top1 acc on the test:{}".format(str(eval_top1_sn(model, test_dataloader, CLASS_NUM, True))))



