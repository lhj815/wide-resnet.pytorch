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
parser.add_argument('--input_preproc_noise_magni', default=None, type=float, help='0.0014')
parser.add_argument('--odin', default=None, type=float, help='1000')
parser.add_argument('--batch_size', default=100, type=int)

args = parser.parse_args()
print(args)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, args.batch_size, cf.optim_type


# Setting_Dir
import datetime
args.outf = str(args.outf)
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

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ]) # meanstd transformation

# transform_test = transforms.Compose([
#     transforms.Resize(32),
#     transforms.ToTensor(),
# ])




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
    nt_test_loader = torch.utils.data.DataLoader(out_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # num_classes = 10
elif (args.out_dataset == 'Tiny'):
    print("| Preparing Tiny_ImageNet dataset for out-of-distrib....")
    sys.stdout.write("| ")
    trainset = None
    out_testset = torchvision.datasets.ImageFolder("../data/Imagenet/", transform=transform_test)
    nt_test_loader = torch.utils.data.DataLoader(out_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # num_classes = 10
elif (args.out_dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset for out-of-distrib.......")
    sys.stdout.write("| ")
    trainset = None
    out_testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    nt_test_loader = torch.utils.data.DataLoader(out_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # num_classes = 100
elif (args.out_dataset == 'LSUN_resize'):
    print("| Preparing LSUN_resize dataset for out-of-distrib....")
    sys.stdout.write("| ")
    trainset = None
    out_testset = torchvision.datasets.ImageFolder("../data/LSUN_resize/", transform=transform_test)
    nt_test_loader = torch.utils.data.DataLoader(out_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # num_classes = 10


# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# #####################################################################
# ################ img extract
# transform_extract = transforms.Compose([
#     transforms.ToTensor(),
# ])
# # extracter = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
# # extract_loader = torch.utils.data.DataLoader(extracter, batch_size=100, shuffle=False, num_workers=2)
# extracter = torchvision.datasets.ImageFolder("../data/Imagenet/", transform=transform_extract)
# extract_loader = torch.utils.data.DataLoader(extracter, batch_size=100, shuffle=True, num_workers=2)
# import cv2

# for batch_idx, (inputs, targets) in enumerate(extract_loader):
#     # inputs = inputs.numpy().transpose(0,2,3,1) * 255
#     inputs = inputs.numpy().transpose(0,2,3,1) * 255
#     for i in range(len(targets)):
#         # print(targets[i].item())
#         cv2.imwrite('imgs/Imagenet/%d_%d.png'%(batch_idx, i), cv2.cvtColor(inputs[i], cv2.COLOR_RGB2BGR))   
#         # cv2.imwrite('imgs/Imagenet/%d_%d.png'%(batch_idx, i), inputs[i])
#         # if targets[i].item() == 23 :
#         #     cv2.imwrite('imgs/sample/%d_%d_%d.png'%(targets[i], batch_idx, i), inputs[i])
#             # cv2.imwrite('imgs/sample/%d_%d_%d.png'%(targets[i], batch_idx, i), cv2.cvtColor(inputs[i], cv2.COLOR_RGB2BGR))
#     if batch_size * batch_idx > 1000 :
#         exit()
        
# exit()
# ######################################################################


# Return network & file name
def getNetwork(args):
    # if args.pretrained == 'wide-resnet':
    #         net = torch.load('./pretrained/wideresnet100.pth')
    #         file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
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
    # checkpoint = torch.load('./checkpoint/{}/{}/{}.pth'.format(args.dataset, args.check_point, os.sep+file_name))
    # checkpoint = torch.load('./checkpoint/{}/{}/{}.pth'.format(args.dataset, args.check_point, os.sep+file_name),map_location='cuda:1')
    
net = checkpoint['net']
epoch = checkpoint['epoch']

# net = torch.load('./pretrained/wideresnet100.pth')['net']
# net = ResNet34(num_c=num_classes)
# net.load_state_dict(torch.load('./pretrained/resnet_cifar100.pth', map_location = "cuda:" + str(0)))
# net = torch.load('./pretrained/resnet_cifar100.pth', map_location = "cuda:" + str(0))

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    # cudnn.benchmark = True

net.eval()

def generate_target():
    net.eval()
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt'%args.outf, 'w')
    f11 = open('%s/confidence_Base_In_correct.txt'%args.outf, 'w')
    f12 = open('%s/confidence_Base_In_wrong.txt'%args.outf, 'w')
    f3 = open('%s/confidence_Base_In_sharing_node.txt'%args.outf, 'w')
    f5 = open('%s/confidence_Base_In_sharing_node_of_correct_case.txt'%args.outf, 'w')
    f7 = open('%s/confidence_Base_In_sharing_node_of_wrong_case.txt'%args.outf, 'w')
    f9 = open('%s/confidence_Base_In_softmax_max_and_sharing_node.txt'%args.outf, 'w')

    for data, target in testloader:
        total += data.size(0)
        #vutils.save_image(data, '%s/target_samples.png'%args.outf, normalize=True)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data, requires_grad = True), Variable(target)
    
        full_output = net(data)
        batch_output = full_output[:,:num_classes]
        
        if args.odin is not None :
            batch_output /= args.odin

        if args.input_preproc_noise_magni is not None :
            # print(args.input_preproc_noise_magni)
            maxIndexTemp = torch.argmax(batch_output, dim=1)
            labels = Variable(maxIndexTemp).cuda()

            # labels = target_transform_for_elementwise_bce(labels, 100)
            # loss = F.binary_cross_entropy_with_logits(batch_output, labels)

            loss = F.cross_entropy(F.softmax(batch_output), labels)
            loss.backward()
            # Normalizing the gradient to binary in {0, 1}
            gradient =  torch.ge(data.grad.data, 0)
            # gradient = (gradient.float())
            gradient = (gradient.float() - 0.5) * 2
            # gradient = data.grad.data
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0]) / 0.2675
            gradient[0][1] = (gradient[0][1]) / 0.2565
            gradient[0][2] = (gradient[0][2]) / 0.2761

            # gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
            # gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
            # gradient[0][2] = (gradient[0][2])/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(data.data,  -args.input_preproc_noise_magni, gradient)
            batch_output = net(Variable(tempInputs))[:,:num_classes]
            
            if args.odin is not None :
                batch_output /= args.odin

        # compute the accuracy
        max_logit, pred = batch_output.data.max(1)
        
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()
        for i in range(data.size(0)):
            # confidence score: max_y p(y|x)
            output = batch_output[i].view(1,-1)
            if args.mode == 'sigmoid':
                soft_out = F.sigmoid(output)
            elif args.mode == 'tanh':
                soft_out = F.tanh(output) / 2.0 + 0.5
            elif args.mode == 'clamp':
                soft_out = torch.clamp(output, min=-10, max=10) / 20.0 + 0.5
            elif args.mode == 'softmax':
                soft_out = F.softmax(output)
            elif args.mode == 'temper':
                soft_out = F.softmax((output - max_logit[i])/100.0)
            elif args.mode == 'sharing':
                output = full_output[i].view(1,-1)
                sharing = output[0,-1].view(1,1)
                one_max_logit = max_logit[i].view(1,1)
                two_logits = torch.cat((one_max_logit, sharing), dim=1)
                soft_out = F.softmax(two_logits)
            elif args.mode == 'sharing_include_softmax' : 
                output = full_output[i].view(1,-1)
                soft_out = F.softmax(output)
                f3.write("{:.4f}\n".format(soft_out[0, num_classes]))
                
                sharing = output[0,-1].view(1,1)
                one_max_logit = max_logit[i].view(1,1)
                two_logits = torch.cat((one_max_logit, sharing), dim=1)
                OOD_node_softmax = F.softmax(two_logits)
                f9.write("{:.4f}\n".format(OOD_node_softmax[0, -1]))
            elif args.mode == 'sharing_include_sigmoid' : 
                output = full_output[i].view(1,-1)
                soft_out = F.sigmoid(output)
                f3.write("{:.4f}\n".format(soft_out[0, num_classes]))
            elif args.mode == 'only_sharing_node':
                soft_out = 1. - F.sigmoid(output[:,-1]).view(1,-1)
                # print(soft_out.size())
                # print(soft_out)
                
                # print(torch.max(soft_out.data))
                # exit()
            elif args.mode == 'sharing_node_print_include_softmax' : 
                output = full_output[i].view(1,-1)
                soft_out = F.softmax(output)
                f9.write("{:.4f}\n".format(torch.max(soft_out[:,:num_classes].data)))
                
                sharing = output[0,-1].view(1,1)
                one_max_logit = max_logit[i].view(1,1)
                two_logits = torch.cat((one_max_logit, sharing), dim=1)
                soft_out = (1. - F.softmax(two_logits))[:,-1]
                f3.write("{:.4f}\n".format(float(soft_out.data.cpu())))
            else :
                assert()
            
            soft_out = torch.max(soft_out.data)
            f1.write("{}\n".format(soft_out))
            if pred[i] != target[i]:
                f12.write("{}\n".format(soft_out))
                # f7.write("{}\n".format(F.sigmoid(batch_output[i,num_classes]).item()))
            else :
                f11.write("{}\n".format(soft_out))
                # f5.write("{}\n".format(F.sigmoid(batch_output[i,num_classes]).item()))
                # f1.write("{}\n".format(F.sigmoid(batch_output[i,num_classes]).item()))
            

    print('\n Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * float(correct) / float(total)))
    return 100. * float(correct) / float(total)

def generate_non_target():
    net.eval()
    total = 0
    f2 = open('%s/confidence_Base_Out.txt'%args.outf, 'w')
    f4 = open('%s/confidence_Base_Out_sharing_node.txt'%args.outf, 'w')
    f6 = open('%s/confidence_Base_Out_what_infer.txt'%args.outf, 'w')
    # f8 = open('%s/confidence_Base_Out_sharing_node.txt'%args.outf, 'w')
    f10 = open('%s/confidence_Base_Out_softmax_max_and_sharing_node.txt'%args.outf, 'w')

    f12 = open('%s/bag_of_out_of_distrib.txt'%args.outf, 'w')
    bags = torch.zeros([num_classes]).cuda()

    for data, target in nt_test_loader:
        total += data.size(0)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        full_output = net(data)
        batch_output = full_output[:,:num_classes]

        if args.odin is not None :
            batch_output /= args.odin

        if args.input_preproc_noise_magni is not None :
            maxIndexTemp = torch.argmax(batch_output, dim=1)
            labels = Variable(maxIndexTemp).cuda()

            # labels = target_transform_for_elementwise_bce(labels, 100)
            # loss = F.binary_cross_entropy_with_logits(batch_output, labels)

            loss = F.cross_entropy(F.softmax(batch_output), labels)

            loss.backward()
            
            # Normalizing the gradient to binary in {0, 1}
            gradient =  torch.ge(data.grad.data, 0)
            # gradient = (gradient.float())
            gradient = (gradient.float() - 0.5) * 2
            # gradient = data.grad.data
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0] )/ 0.2675
            gradient[0][1] = (gradient[0][1] )/ 0.2565
            gradient[0][2] = (gradient[0][2])/ 0.2761
            # gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
            # gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
            # gradient[0][2] = (gradient[0][2])/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(data.data,  -args.input_preproc_noise_magni, gradient)
            batch_output = net(Variable(tempInputs))[:,:num_classes]

            if args.odin is not None :
                batch_output /= args.odin


        max_logit, pred = batch_output.data.max(1)
        bags += count(pred, num_classes)
        for i in range(data.size(0)):
            # confidence score: max_y p(y|x)
            output = batch_output[i].view(1,-1)
            if args.mode == 'sigmoid':
                soft_out = F.sigmoid(output)
            elif args.mode == 'tanh':
                soft_out = F.tanh(output) / 2.0 + 0.5
            elif args.mode == 'clamp':
                soft_out = torch.clamp(output, min=-10, max=10) / 20.0 + 0.5
            elif args.mode == 'softmax':
                soft_out = F.softmax(output)
            elif args.mode == 'temper':
                soft_out = F.softmax((output - max_logit[i])/100.0)
            elif args.mode == 'sharing':
                output = full_output[i].view(1,-1)
                sharing = output[0,-1].view(1,1)
                one_max_logit = max_logit[i].view(1,1)
                two_logits = torch.cat((one_max_logit, sharing), dim=1)
                soft_out = F.softmax(two_logits)
            elif args.mode == 'sharing_include_softmax' : 
                output = full_output[i].view(1,-1)
                soft_out = F.softmax(output)
                f4.write("{:.4f}\n".format(soft_out[0, num_classes]))

                sharing = output[0,-1].view(1,1)
                one_max_logit = max_logit[i].view(1,1)
                two_logits = torch.cat((one_max_logit, sharing), dim=1)
                OOD_node_softmax = F.softmax(two_logits)
                f10.write("{:.4f}\n".format(OOD_node_softmax[0, -1]))
            elif args.mode == 'sharing_include_sigmoid' : 
                output = full_output[i].view(1,-1)
                soft_out = F.sigmoid(output)
                f4.write("{:.4f}\n".format(soft_out[0, num_classes]))
            elif args.mode == 'only_sharing_node':
                soft_out = 1. - F.sigmoid(output[:,-1]).view(1,-1)
            elif args.mode == 'sharing_node_print_include_softmax' : 
                output = full_output[i].view(1,-1)
                soft_out = F.softmax(output)
                f10.write("{:.4f}\n".format(torch.max(soft_out[:,:num_classes].data)))
                
                sharing = output[0,-1].view(1,1)
                one_max_logit = max_logit[i].view(1,1)
                two_logits = torch.cat((one_max_logit, sharing), dim=1)
                soft_out = (1. - F.softmax(two_logits))[:,-1]
                f4.write("{:.4f}\n".format(float(soft_out.data.cpu())))
            else :
                assert()
            # print(soft_out.sum())
            
            soft_out = torch.max(soft_out.data)
            # if pred[i] == 2 or pred[i] == 3 or pred[i] ==5:
            #     pass
            # else :
            f2.write("{}\n".format(soft_out))
            # f2.write("{}\n".format(F.sigmoid(batch_output[i,num_classes]).item()))
            # if soft_out > 0.9 :
            f6.write("{}\n".format(pred[i]))
            # f8.write("{}\n".format(F.sigmoid(batch_output[i,num_classes]).item()))
        
    f12.write("{}".format(bags.clone().cpu().data.detach().numpy()))

    if args.out_dataset == 'cifar100':
        if args.mode == 'sigmoid':
            mode = 'bce'
        elif args.mode == 'softmax':
            mode = 'ce'
        write_output(batch_output, target, args.outf, 0, num_classes=100, mode=mode, cifar100=True)
        
print('generate log from in-distribution data')
acc = generate_target()
print('generate log  from out-of-distribution data')
generate_non_target()
print('calculate metrics')
callog.metric(args.outf, acc)
