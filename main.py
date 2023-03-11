import pdb

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

from model import ServeNet
from utils import load_data_test, load_data_train, eval_top1, eval_top5
import util

# create working servenet model
# change the dataloader to index dataloader
# similarity matrix from gradients
# get subset from the similarity matrix, I mean it would be same as that of craig on CIFAR100.

from hyper_param import *


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
        self.data_set = load_data_train()

    def __getitem__(self, index):
        data = self.data_set[index]
        # Your transformations here (or set it in CIFAR10)
        return data, index

    def __len__(self):
        return len(self.data_set)

if __name__ == "__main__":
    train_data = load_data_train()
    test_data = load_data_test()

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



    model = ServeNet(768, CLASS_NUM)

    model.weight_sum.w1 = torch.nn.Parameter(torch.tensor([0.5]))
    model.weight_sum.w2 = torch.nn.Parameter(torch.tensor([0.5]))

    # model.bert_description.requires_grad_(False)
    # model.bert_name.requires_grad_(False)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()

    # pdb.set_trace()

    epoch_list = []
    acc1_list = []
    acc5_list = []
    best_accuracy = 0

    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params_all = sum(p.numel() for p in model.parameters())
    print("Trainable: ", pytorch_total_params_trainable)
    print("All: ", pytorch_total_params_all)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    w11 = model.module.weight_sum.w1
    w22 = model.module.weight_sum.w2
    print("w11: ", w11)
    print("w22: ", w22)

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

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        # pdb.set_trace()
        top_1_acc = eval_top1(model, test_dataloader, category_num)
        top_5_acc = eval_top5(model, test_dataloader)
        epoch_list.append(epoch + 1)
        acc1_list.append(top_1_acc)
        acc5_list.append(top_5_acc)

        if top_1_acc > best_accuracy:
            best_accuracy = top_1_acc
            torch.save(model, f"./files/snlt3_best3_{CLASS_NUM}_{int(SUBSET_SIZE*100)}b_full")

        print("=======>top1 acc on the test:{}".format(str(top_1_acc)))
        print("=======>top5 acc on the test:{}".format(str(top_5_acc)))
        w11 = model.module.weight_sum.w1
        w22 = model.module.weight_sum.w2
        print("w11: ", w11)
        print("w22: ", w22)

        acc_list = pd.DataFrame(
            {
                'epoch': epoch_list,
                'Top1': acc1_list,
                'Top5': acc5_list
            }
        )

        acc_list.to_csv(f'./files/t3_SN_{CLASS_NUM}_{int(SUBSET_SIZE*100)}b_full.csv')

        if SUBSET_SIZE < 1:
            np.savez(f"./files/subset3_{int(SUBSET_SIZE*100)}b_{CLASS_NUM}c_full",
                     subset=selected_ndx, weight=selected_wgt)

    # print("=======>top1 acc on the test:{}".format(str(eval_top1_sn(model, test_dataloader, CLASS_NUM, True))))



