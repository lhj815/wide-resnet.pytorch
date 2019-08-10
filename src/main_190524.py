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
parser.add_argument('--inferOnly', '-infer', action='store_true', help='inference mode with the saved model (without accuracy)')
parser.add_argument('--loss', '-l', default='ce', help='ce / bce / bce_and_ce ')
parser.add_argument('--ent', type=float, default=0.0, help='entropy_maximization weight beta')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

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
    out_testloader = torch.utils.data.DataLoader(out_testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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

# ######################################################################
# ################# img extract
# transform_extract = transforms.Compose([
#     transforms.ToTensor(),
# ])
# extracter = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_extract)
# extract_loader = torch.utils.data.DataLoader(extracter, batch_size=100, shuffle=False, num_workers=2)
# import cv2
# for batch_idx, (inputs, targets) in enumerate(extract_loader):
#     inputs = inputs.numpy().transpose(0,2,3,1) * 255
#     for i in range(len(targets)):
#         # if targets[i] == 23 :
#         cv2.imwrite('imgs/sample/%d_%d_%d.png'%(targets[i], batch_idx, i), cv2.cvtColor(inputs[i], cv2.COLOR_RGB2BGR))
# exit()
# #######################################################################

# infer only option
if (args.inferOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    epoch = checkpoint['epoch']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    total = 0
    total_out = 0
    
    confi_list = []
    confi_list_out = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        if args.loss == 'bce':
            outputs = F.sigmoid(net(inputs))
        elif args.loss == 'ce' :
            outputs = F.softmax(net(inputs))
        else :
            outputs = net(inputs)
        pred_confi, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

    
    for batch_idx, (inputs, targets) in enumerate(out_testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        if args.loss == 'bce':
            outputs = F.sigmoid(net(inputs))
        elif args.loss == 'ce' :
            outputs = F.softmax(net(inputs))
        else :
            outputs = net(inputs)
        pred_confi_out, predicted_out = torch.max(outputs.data, 1)
        total_out += targets.size(0)



    print('finish')

    sys.exit(0)


# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()


    test_loss = 0
    temp1_accum = 0
    temp2_accum = 0
    correct = 0
    total = 0
    false_bags = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        temp1_accum = loss.detach().cpu() * (1./(batch_idx+1.)) + temp1_accum * (batch_idx/(batch_idx+1.))
        if 'bce' in args.loss :
            bce_targets = target_transform_for_elementwise_bce(targets, outputs).cuda()
            temp2 = criterion2(F.sigmoid(outputs), bce_targets)
            temp2_accum = temp2.detach().cpu() * (1./(batch_idx+1.)) + temp2_accum *  (batch_idx/(batch_idx+1.))
            loss += temp2

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        wrong = predicted.ne(targets)

        for i in range(len(wrong)):
                if wrong[i] == 1:
                    temp = []
                    temp.append(predicted[i].detach().cpu().numpy().max())
                    
                    temp.append(targets[i].detach().cpu().numpy().max())
                    temp.append(outputs[i].detach().cpu().numpy().max())
                    temp.append(outputs[i][targets[i].detach().cpu().numpy()].detach().cpu().numpy().max())
                    temp.append(outputs[i].detach().cpu().numpy())
                    # print(i, predicted[i].detach().cpu().numpy(), targets[i].detach().cpu().numpy(), outputs[i].detach().cpu().numpy())
                    false_bags.append(temp)

    acc = 100.*correct/total

    false_bags = [acc] + false_bags
    write_list(false_bags, 'bce', epoch)

    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print("| total loss:%.4f  CE_loss:%.4f  BCE_loss:%.4f"%(temp1_accum+temp2_accum, temp1_accum, temp2_accum))

    sys.exit(0)

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

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()



# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    temp1_accum = 0
    temp2_accum = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), nesterov=True,momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        temp1_accum = loss.detach().cpu() * (1./(batch_idx+1.)) + temp1_accum * (batch_idx/(batch_idx+1.))
        if args.loss == 'bce' :
            loss = torch.zeros(1).cuda()
        if 'bce' in args.loss :
            bce_targets = target_transform_for_elementwise_bce(targets, num_classes).cuda()
            if num_classes > 30 :
                new_outputs, new_targets = sampling_for_loss(outputs, targets)
                temp2 = criterion2(F.sigmoid(new_outputs), new_targets).cuda()
            else:
                temp2 = criterion2(F.sigmoid(outputs), bce_targets)
            temp2_accum = temp2.detach().cpu() * (1./(batch_idx+1.)) + temp2_accum *  (batch_idx/(batch_idx+1.))
            
            loss += temp2
            loss = loss
            
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f CE_loss : %.4f BCE_loss : %4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), temp1_accum, temp2_accum, 100.*correct/total))
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
        loss = criterion(outputs, targets)
        temp1_accum = loss.detach().cpu() * (1./(batch_idx+1.)) + temp1_accum * (batch_idx/(batch_idx+1.))
        if 'bce' in args.loss :
            bce_targets = target_transform_for_elementwise_bce(targets, num_class=num_classes).cuda()
            temp2 = criterion2(F.sigmoid(outputs), bce_targets)
            temp2_accum = temp2.detach().cpu() * (1./(batch_idx+1.)) + temp2_accum *  (batch_idx/(batch_idx+1.))
            loss += temp2
        

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f  CE_loss : %.4f  BCE_loss : %.4f  Acc@1: %.2f%%" 
        %(epoch, loss.item(), temp1_accum, temp2_accum, acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+args.dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
