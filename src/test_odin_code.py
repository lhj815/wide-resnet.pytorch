from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable

import random
from utils import *
import calculate_log as callog

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--out_dataset', default='SVHN', type=str, help='dataset = [cifar10/cifar100/SVHN/Tiny]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--inferOnly', '-infer', action='store_true', help='inference mode with the saved model (without accuracy)')
parser.add_argument('--loss', '-l', default='ce', help='ce / bce / bce_and_ce / temper')
parser.add_argument('--mode', default='sigmoid', help='sigmoid / softmax / etc')
parser.add_argument('--outf', default='./results/test_detection/', help='folder to output images and model checkpoints')
parser.add_argument('--check_point', default=None)

args = parser.parse_args()
print(args)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type


# Setting_Dir
import datetime
args.outf = str(args.outf) + str(args.dataset) + '/' + str(args.out_dataset) +'_'+ str(datetime.datetime.now())[:10] + '_' + \
    str(datetime.datetime.now())[11:19] + '_' + str(args.mode)
os.makedirs(args.outf)



# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])



if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 100
if (args.out_dataset == 'SVHN'):
    print("| Preparing SVHN dataset for out-of-distrib....")
    sys.stdout.write("| ")
    # trainset = torchvision.datasets.SVHN(root='./data', train=True, download=True, transform=transform_train)
    trainset = None
    out_testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
    nt_test_loader = torch.utils.data.DataLoader(out_testset, batch_size=100, shuffle=False, num_workers=2)
    # num_classes = 10
elif (args.out_dataset == 'Tiny'):
    print("| Preparing Tiny_ImageNet dataset for out-of-distrib....")
    sys.stdout.write("| ")
    trainset = None
    out_testset = torchvision.datasets.ImageFolder("../data/imagenet-200/test/", transform=transform_test)
    nt_test_loader = torch.utils.data.DataLoader(out_testset, batch_size=100, shuffle=False, num_workers=2)
    # num_classes = 10



trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# #####################################################################
# ################ img extract
# transform_extract = transforms.Compose([
#     transforms.ToTensor(),
# ])
# extracter = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
# extract_loader = torch.utils.data.DataLoader(extracter, batch_size=100, shuffle=False, num_workers=2)
# # extracter = torchvision.datasets.ImageFolder("../data/imagenet-200/test/", transform=transform_extract)
# # extract_loader = torch.utils.data.DataLoader(extracter, batch_size=100, shuffle=False, num_workers=2)
# import cv2
# for batch_idx, (inputs, targets) in enumerate(extract_loader):
#     inputs = inputs.numpy().transpose(0,2,3,1) * 255
#     for i in range(len(targets)):
#         print(targets[i].item())
#         if targets[i].item() == 23 :
#             cv2.imwrite('imgs/sample/%d_%d_%d.png'%(targets[i], batch_idx, i), inputs[i])
#             # cv2.imwrite('imgs/sample/%d_%d_%d.png'%(targets[i], batch_idx, i), cv2.cvtColor(inputs[i], cv2.COLOR_RGB2BGR))
# exit()
# ######################################################################


# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, num_classes, args.widen_factor, dropRate=args.dropout)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name



assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
_, file_name = getNetwork(args)
if args.check_point is None :
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
else :
    checkpoint = torch.load('./checkpoint/{}/{}/{}.t7'.format(args.dataset, args.check_point, os.sep+file_name))
net = checkpoint['net']
epoch = checkpoint['epoch']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

net.eval()

from odin import calMetric as m
from odin import calData as d
import torch.backends.cudnn as cudnn


def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    d.testData(net, criterion, CUDA_DEVICE, testloader, testloaderOut, nnName, dataName, epsilon, temperature) 
    m.metric(nnName, dataName)