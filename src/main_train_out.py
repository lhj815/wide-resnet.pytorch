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
from calculate_log import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--out_dataset', default=None, type=str, help='dataset = [cifar10/cifar100/SVHN]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--inferOnly', '-infer', default=False)# action='store_true', help='inference mode with the saved model (without accuracy)')
parser.add_argument('--loss', '-l', default='ce', help='ce / bce / bce_and_ce ')
parser.add_argument('--bce_scale', type=float , default=1.)
parser.add_argument('--ent', type=float, default=0.0, help='entropy_maximization weight beta')
parser.add_argument('--sharing', type=float, default=None, help='weight of CE between sharing node and target node')
parser.add_argument('--unknown_is_True', default=False, help='add_concept of unknown')
parser.add_argument('--sepa_unknown_sharing', default=False, help='if true -> separation : unknown node & sharing node / e.g cifar10 : true=>12 / false=>11')
parser.add_argument('--sampling_rate', type=float , default=1.)
parser.add_argument('--sigmoid_sum', type=float , default=None)
parser.add_argument('--out_folder', type=str, default=None)
parser.add_argument('--pretrained', type=str, default=None, help='resnet50 / resnet34 / wide-resnet')
# parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
# torch.cuda.set_device(args.gpu)
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, args.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
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
    out_testloader = torch.utils.data.DataLoader(out_testset, batch_size=100, shuffle=False, num_workers=0)
    num_classes = 10
elif(args.out_dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    out_testset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    out_testloader = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 100
elif (args.out_dataset == 'Tiny'):
    print("| Preparing Tiny_ImageNet dataset for out-of-distrib....")
    sys.stdout.write("| ")
    out_testset = torchvision.datasets.ImageFolder("../data/Imagenet_random_sample_1k/", transform=transform_test)
    out_testloader = torch.utils.data.DataLoader(out_testset, batch_size=batch_size, shuffle=False, num_workers=2)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

args.num_classes = num_classes

# Return network & file name
def getNetwork(args):
    if args.pretrained is not None :
        print('using pretrained :', args.pretrained)
        if args.pretrained == 'resnet34':
            net = torchvision.models.resnet34(pretrained=True)
            file_name = 'resnet34'    
            # for param in net.parameters():
            #     param.requires_grad = False
            # Replace the last fully-connected layer
            # Parameters of newly constructed modules have requires_grad=True by default
            net.fc = nn.Linear(512, num_classes) # assuming that the fc7 layer has 512 neurons, otherwise change it
        elif args.pretrained == 'wide-resnet':
            net = torch.load('./pretrained/wideresnet100.pth')
            file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
        return net, file_name
    
    else:
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

def write_tensor(something, name='sigmoid', epoch=None):
    with open('./results/outputs/outputs_{}_{}.txt'.format(name, epoch), 'w') as f:
        for i in range(len(something)):
            print(something[i].type)
            if something[i].type():
                f.write(str(something[i].detach().cpu().numpy()))
                f.write('\n')
        f.close()

def write_list(something, name='sigmoid', epoch=None):
    with open('./results/outputs/outputs_{}_{}.txt'.format(name, epoch), 'w') as f:
        f.write('acc : %f \n\n'%something[0])
        f.write('infer / target / confi_of_infer / confi_of_target / confi_all\n')
        for i in range(1, len(something)):
            f.write(str(something[i]))
            f.write('\n\n')
        f.close()

if args.sharing is not None :
    num_classes += 1
    if args.sepa_unknown_sharing :
        num_classes += 1
    print(num_classes)


# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    # net.apply(conv_init)

if args.sharing is not None :
    num_classes += -1
    if args.sepa_unknown_sharing :
        num_classes += -1


if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(1))
    # cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()
entropy = HLoss()
sharing_entropy = HLoss_for_3d_tensor()


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_classes = args.num_classes
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), nesterov=True,momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(args.lr, epoch), betas=(0.5,0.999), weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, _) in enumerate(out_testloader):
        if use_cuda:
            inputs = inputs.cuda() # GPU settings
        optimizer.zero_grad()
        inputs = Variable(inputs)
        outputs = net(inputs)               # Forward Propagation

        # targets = torch.ones_like(outputs[:,-1]).cuda()
        targets = torch.zeros_like(outputs).cuda()
        targets[:,-1] = 1

        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        loss.backward()

        optimizer.step()
        
        max_logit, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(10).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                        %(epoch, num_epochs, batch_idx+1, (len(out_testset)//batch_size)+1, 
                        loss.item(), float(100.00*float(correct)/float(total))))        
        sys.stdout.flush()
        
        

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    temp1_accum = 0
    temp2_accum = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs[:,:num_classes], targets)
        temp1_accum = loss.detach().cpu() * (1./(batch_idx+1.)) + temp1_accum * (batch_idx/(batch_idx+1.))
        if 'bce' in args.loss :
            bce_targets = target_transform_for_elementwise_bce(targets, num_class=num_classes).cuda()
            temp2 = criterion2(F.sigmoid(outputs[:,:num_classes]), bce_targets)
            temp2_accum = temp2.detach().cpu() * (1./(batch_idx+1.)) + temp2_accum *  (batch_idx/(batch_idx+1.))
            loss += temp2
        test_loss += loss.item()
        _, predicted = torch.max(outputs[:,:num_classes].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        # if epoch % 4 == 0:
        # new_outputs, new_targets = sampling_for_loss(outputs[:,:num_classes], targets)
        # if args.loss == 'bce':
        #     write_output(new_outputs, new_targets.long(), './checkpoint/{}'.format(args.dataset), epoch, args.loss)
        # else :
            # write_output(outputs, targets, './checkpoint/{}'.format(args.dataset), epoch, args.loss)
        
        write_output(outputs, targets, args.out_folder, epoch, num_classes ,args.loss)
    # Save checkpoint when best model
    acc = float(100.00 * float(correct)/float(total))
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f  CE_loss : %.4f  BCE_loss : %.4f  Acc@1: %.2f%%" 
        %(epoch, loss.item(), temp1_accum, temp2_accum, acc))

    print('| Saving model...\t\t\tTop1 = %.2f%%' %(acc), '{}epoch: '.format(epoch))
    state = {
            'net':net.module if use_cuda else net,
            'acc':acc,
            'epoch':epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = args.out_folder
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    torch.save(state, save_point+file_name+'_{}epoch.t7'.format(str(epoch)))



print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()
    if args.inferOnly :
        testloader = out_testloader
        test(epoch)
        exit()
        
    train(epoch)
    test(epoch)
    
    # exit()

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))



print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
