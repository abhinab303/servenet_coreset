print(__doc__)
import matplotlib
#matplotlib.use('TkAgg')

import heapq
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy import spatial
import matplotlib.pyplot as plt

import pdb


class FacilityLocation:

    def __init__(self, D, V, alpha=1.):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

        self.old_max = None
        self.before_diff = None

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)

            sum_term = self.f_norm * np.maximum(self.curMax, self.D[:, ndx])
            # return self.norm * math.log(1 + sum_term.sum() + np.sqrt(sum_term.var())) - self.curVal            # Variance
            return self.norm * math.log(1 + sum_term.sum()) - self.curVal                               # Original
        else:
            sum_term = self.f_norm * self.D[:, ndx]
            # return self.norm * math.log(1 + sum_term.sum() + np.sqrt(sum_term.var())) - self.curVal            # Variance
            return self.norm * math.log(1 + sum_term.sum()) - self.curVal                               # Original

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.old_max = self.curMax
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]

        sum_term = self.f_norm * self.curMax
        # self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.curVal = self.norm * math.log(1 + sum_term.sum())                                          # original
        # self.curVal = self.norm * math.log(1 + sum_term.sum() + np.sqrt(sum_term.var()))                       # variance
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B, testp=None):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)           # gives the max value when popping, max value at root of heap
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    if testp:
        print("Similarity: \n", F.D)
        print("Heap: ", sorted(order, key=lambda x: x[0], reverse=True))
        print("........")
        print("........")


    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements

        if improv >= 0:
            if not order:
                # pdb.set_trace()
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

        # pdb.set_trace()
        if testp:
            print("CurrentEl: ", el[1])
            print("Heap: ", sorted(order, key=lambda x: x[0], reverse=True))
            print("CurMax: ", F.curMax)
            # print("MaxVal: ", F.norm * math.log(1 + F.f_norm * F.old_max.sum()) if type(F.old_max) != type(None) else 0 )
            # print("Curr val: ", improv + F.norm * math.log(1 + F.f_norm * F.old_max.sum()) if type(F.old_max) != type(None) else 0 )
            print("Curr inc: ", improv, el[1])
            print("compare to: ", heapq.nlargest(1, order)[0] if order else None)

            print("sset: ", sset)
            # print("compare to: ", heapq.nlargest(2, order)[1] if len(order) > 1 else None)
            print(".............")
            print(".............")
        F.old_max = F.curMax

    # pdb.set_trace()
    # print("len ss: ", len(sset), sset)
    return sset, vals


def test():
    n = 5
    np.random.seed(0)
    X = np.random.rand(n, n)
    D = X * np.transpose(X)
    # D = np.array([[4.49061341e-04, 4.98258410e-01, 4.20882328e-04, 3.87854993e-01],
    #             [4.98258410e-01, 2.94626449e-01, 6.93003033e-01, 7.51619541e-02],
    #             [4.20882328e-04, 6.93003033e-01, 3.15659677e-02, 1.41936316e-01],
    #             [3.87854993e-01, 7.51619541e-02, 1.41936316e-01, 4.71564485e-05]])
    F = FacilityLocation(D, [x for x in range(0, n)])
    sset = lazy_greedy_heap(F, [x for x in range(0, n)], 15, test=True)
    print(sset)

if __name__ == '__main__':
    test()

    ######
    ### python train_resnet.py -s 0.1 -w -b 128 -g --smtk 0
    ### run: 0, subset_size: 0.1, epoch: 199, prec1: 58.23, loss: 1.128, g: 0.100, b: 0.100, best_prec1_gb: 74.33,
    ### best_loss_gb: 0.744, best_run: 0;  best_prec_all: 74.33, best_loss_all: 0.744, best_g: 0.100, best_b: 0.100,
    ### not selected %:0.034

    ######
    ### python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0
    ### FL time: 3.024, Sim time: 0.117
    ### 0.768 % not selected yet
    ### * Prec@1 83.570
    ### * Prec@1 86.788
    ### run: 0, subset_size: 0.1, epoch: 199, prec1: 83.56999987792969, loss: 0.386, g: 0.100, b: 0.100,
    ### best_prec1_gb: 83.62999992675782, best_loss_gb: 0.386, best_run: 0;  best_prec_all: 83.62999992675782,
    ### best_loss_all: 0.386, best_g: 0.100, best_b: 0.100, not selected %:0.768
    ### Saving the results to /tmp/cifar10_sgd_moment_0.9_resnet20_0.1_grd_w_warm_mile_start_0_lag_1_var



