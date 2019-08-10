## original code is from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
## Modeified by Kimin Lee
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

def tpr95(dir_name):
    #calculate the falsepositive error when tpr is 95%
    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/200000 # precision:200000

    total = 0.0
    fpr = 0.0
    thresh = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            thresh += delta
            fpr += error2
            total += 1
    if total == 0:
        print('corner case')
        fprBase = 1
    else:
        fprBase = fpr/total
        thresh = thresh/total

    return fprBase, thresh


def auroc(dir_name):
    #calculate the AUROC
    f1 = open('%s/Update_Base_ROC_tpr.txt'%dir_name, 'w')
    f2 = open('%s/Update_Base_ROC_fpr.txt'%dir_name, 'w')

    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/200000

    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        f1.write("{}\n".format(tpr))
        f2.write("{}\n".format(fpr))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr

    return aurocBase

def auprIn(dir_name):
    #calculate the AUPR

    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    precisionVec = []
    recallVec = []
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/200000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def auprOut(dir_name):
    #calculate the AUPR
    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/200000

    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def detection(dir_name):
    #calculate the minimum detection error

    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    end = np.max([np.max(X1), np.max(Y1)])
    start = np.min([np.min(X1),np.min(Y1)])
    gap = (end- start)/200000

    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        temp = errorBase
        errorBase = np.minimum(errorBase, (tpr+error2)/2.0)
        if temp != errorBase :
            threshold = delta

    

    return errorBase, threshold

def detection_at_thresh_50percent(dir_name):
    #calculate the minimum detection error

    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar

    tpr = np.sum(np.sum(X1 < 0.5)) / np.float(len(X1))
    error2 = np.sum(np.sum(Y1 > 0.5)) / np.float(len(Y1))
    errorBase = (tpr+error2)/2.0

    return errorBase

import matplotlib.pyplot as plt
def draw_graph(dir_name, acc):
    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    bins = np.arange(0,1.005, step=0.01)
    # bins = [0.0, 0.05, 0.1,0.15, 0.2, 0.25,0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9,0.95, 1.0]
    plt.hist(X1, bins=bins, alpha=0.5, color='black', label='in', rwidth= 1.4, weights=np.ones(len(X1)) / len(X1))
    plt.hist(Y1, bins=bins, alpha=0.5, color='orange', label='out', rwidth=1.4, weights=np.ones(len(Y1)) / len(Y1))
    plt.ylim([0,1])
    xticks = np.arange(0, 1.005, step=0.1)
    plt.xticks(xticks)
    plt.axvline(X1.mean(), color='black', linestyle='dashed', linewidth=1)
    plt.axvline(Y1.mean(), color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(acc/100., color='red', linestyle='dashed', linewidth=2, label='acc:{:.2f}%'.format(acc))
    plt.legend(loc='upper right')
    plt.savefig('{}/graph_{}acc.png'.format(dir_name,acc))
    plt.close()

    cifar_sharing_node_correct = np.loadtxt('%s/confidence_Base_In_sharing_node_of_correct_case.txt'%dir_name, delimiter=',')
    cifar_sharing_node__wrong = np.loadtxt('%s/confidence_Base_In_sharing_node_of_wrong_case.txt'%dir_name, delimiter=',')
    other_sharing_node = np.loadtxt('%s/confidence_Base_Out_sharing_node.txt'%dir_name, delimiter=',')
    Y1_sharing = other_sharing_node
    X1_sharing = cifar_sharing_node_correct
    X2_sharing = cifar_sharing_node__wrong
    bins = np.arange(0,1.005, step=0.02)
    # bins = [0.0, 0.05, 0.1,0.15, 0.2, 0.25,0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9,0.95, 1.0]
    plt.hist(X1_sharing, bins=bins, alpha=0.3, color='black', label='in_right_sharing', rwidth= 0.7, weights=np.ones(len(X1_sharing)) / len(X1_sharing))
    plt.hist(X2_sharing, bins=bins, alpha=0.3, color='green', label='in_wrong_sharing', rwidth= 0.7, weights=np.ones(len(X2_sharing)) / len(X2_sharing))
    plt.hist(Y1_sharing, bins=bins, alpha=0.3, color='orange', label='out_sharing', rwidth=0.7, weights=np.ones(len(Y1_sharing)) / len(Y1_sharing))
    plt.ylim([0,1])
    plt.xticks(xticks)
    plt.legend(loc='upper right')
    plt.savefig('{}/graph_{}acc_OOD_node_split.png'.format(dir_name,acc))
    plt.close()


    cifar_sharing_node = np.loadtxt('%s/confidence_Base_In_sharing_node.txt'%dir_name, delimiter=',')
    other_sharing_node = np.loadtxt('%s/confidence_Base_Out_sharing_node.txt'%dir_name, delimiter=',')
    Y1_sharing = other_sharing_node
    X1_sharing = cifar_sharing_node
    # bins = [0.0, 0.05, 0.1,0.15, 0.2, 0.25,0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9,0.95, 1.0]
    # bins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    bins = np.arange(0,0.2, step=0.0033)
    # bins = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525]
    plt.hist(X1_sharing, bins=bins, alpha=0.3, color='black', label='in_OOD_node', rwidth= 0.7, weights=np.ones(len(X1_sharing)) / len(X1_sharing))
    plt.hist(Y1_sharing, bins=bins, alpha=0.3, color='orange', label='out_OOD_node', rwidth=0.7, weights=np.ones(len(Y1_sharing)) / len(Y1_sharing))
    plt.ylim([0,1])
    plt.legend(loc='upper right')
    plt.savefig('{}/graph_{}acc_OOD_node.png'.format(dir_name,acc))
    plt.close()


    cifar_sharing_node = np.loadtxt('%s/confidence_Base_In_softmax_max_and_sharing_node.txt'%dir_name, delimiter=',')
    other_sharing_node = np.loadtxt('%s/confidence_Base_Out_softmax_max_and_sharing_node.txt'%dir_name, delimiter=',')
    X1_sharing_soft = cifar_sharing_node
    Y1_sharing_soft = other_sharing_node
    bins = np.arange(0,0.6, step=0.01)
    # bins = [0.0, 0.05, 0.1,0.15, 0.2, 0.25,0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9,0.95, 1.0]
    # bins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    # bins = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525]
    plt.hist(X1_sharing_soft, bins=bins, alpha=0.3, color='black', label='in_soft(max,OOD_node)', rwidth= 0.7, weights=np.ones(len(X1_sharing)) / len(X1_sharing))
    plt.hist(Y1_sharing_soft, bins=bins, alpha=0.3, color='red', label='out_soft(max,OOD_node)', rwidth=0.7, weights=np.ones(len(Y1_sharing)) / len(Y1_sharing))
    plt.ylim([0,1])
    plt.legend(loc='upper right')
    plt.savefig('{}/graph_{}acc_soft_max_and_OOD_node.png'.format(dir_name,acc))
    plt.close()
    
    

    cifar_correct = np.loadtxt('%s/confidence_Base_In_correct.txt'%dir_name, delimiter=',')
    cifar_wrong = np.loadtxt('%s/confidence_Base_In_wrong.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1_correcdt = cifar_correct
    X1_wrong = cifar_wrong
    # bins = [0.0, 0.05, 0.1,0.15, 0.2, 0.25,0.3, 0.35,0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9,0.95, 1.0]
    bins = np.arange(0,1.005, step=0.02)
    plt.hist(X1_correcdt, bins=bins, alpha=0.3, color='black', label='in_right', rwidth= 0.7, weights=np.ones(len(X1_correcdt)) / len(X1_correcdt))
    plt.hist(X1_wrong, bins=bins, alpha=0.3, color='green', label='in_wrong', rwidth= 0.7, weights=np.ones(len(X1_wrong)) / len(X1_wrong))
    plt.hist(Y1, bins=bins, alpha=0.3, color='orange', label='out', rwidth=0.7, weights=np.ones(len(Y1)) / len(Y1))
    plt.ylim([0,1])
    plt.xticks(xticks)
    plt.axvline(X1_correcdt.mean(), color='black', linestyle='dashed', linewidth=1)
    plt.axvline(X1_wrong.mean(), color='green', linestyle='dashed', linewidth=1)
    plt.axvline(Y1.mean(), color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(acc/100., color='red', linestyle='dashed', linewidth=2, label='acc:{:.1f}%'.format(acc))
    plt.legend(loc='upper right')
    plt.savefig('{}/graph_{}acc_right_wrong_split.png'.format(dir_name,acc))
    plt.close()



def metric(dir_name, acc=None, mode='all'):
    draw_graph(dir_name, acc)
    if mode == 'graph':
        print('graph draw_finish -> exit')
        exit()
    print("{:>34}".format("Performance of Baseline detector"))
    fprBase, thresh = tpr95(dir_name)
    print("{:20}{:13.3f}%".format("TNR at TPR 95%:",(1-fprBase)*100))
    print('threshold at TPR 95% : {}'.format(thresh))
    

    errorBase, threshold = detection(dir_name)
    print("{:20}{:13.3f}%".format("Detection acc:",(1-errorBase)*100))
    print('threshold at min(detection error) : {}'.format(threshold))
    errorBase_at_thresh_50percent = detection_at_thresh_50percent(dir_name)
    print('Detection acc at 0.5 threshold : {}'.format((1-errorBase_at_thresh_50percent) * 100.))
    
    aurocBase = auroc(dir_name)
    print("{:20}{:13.3f}%".format("AUROC:",aurocBase*100))

    auprinBase = auprIn(dir_name)
    print("{:20}{:13.3f}%".format("AUPR In:",auprinBase*100))
    auproutBase = auprOut(dir_name)
    print("{:20}{:13.3f}%".format("AUPR Out:",auproutBase*100))

    with open('{}/result.txt'.format(dir_name), 'w') as f :
        f.write("{:>34}\n".format("Performance of Baseline detector"))
        f.write("{:20}{:13.3f}%\n".format("TNR at TPR 95%:",(1-fprBase)*100))
        f.write("{:20}{:13.3f}\n".format("Threshold at TPR 95%:", thresh))
        f.write("{:20}{:13.3f}%\n".format("Detection acc:",(1-errorBase)*100))
        f.write("{:20}{:13.3f}\n".format("Threshold at Detectir acc.", threshold))
        f.write("{:20}{:13.3f}\n".format("Detectir acc at 0.5 thresh.", (1-errorBase_at_thresh_50percent)*100.))
        f.write("{:20}{:13.3f}%\n".format("AUROC:",aurocBase*100))
        f.write("{:20}{:13.3f}%\n".format("AUPR In:",auprinBase*100))
        f.write("{:20}{:13.3f}%\n".format("AUPR Out:",auproutBase*100))

