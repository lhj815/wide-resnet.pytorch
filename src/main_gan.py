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
parser.add_argument('--gan', default=True)
parser.add_argument('--fake_node_bce_beta', default=0.5, type=float)

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
    out_testloader = torch.utils.data.DataLoader(out_testset, batch_size=100, shuffle=False, num_workers=0)
    num_classes = 10
elif(args.out_dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    out_testset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    out_testloader = torchvision.datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
    num_classes = 100   

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

args.num_classes = num_classes

# Return network & file name
def getNetwork(args):
    if args.pretrained is not None :
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

if args.gan :
    print('load GAN')
    nz = 100
    netG = Generator(1, nz, 64, 3) # ngpu, nz, ngf, nc
    netD = Discriminator(1, 3, 64) # ngpu, nc, ndf
    # Initial setup for GAN
    real_label = 1
    fake_label = 0

    gan_criterion = nn.BCELoss()
    fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise)

    

    if use_cuda :
        netD.cuda()
        netG.cuda()
        gan_criterion.cuda()
        fixed_noise = fixed_noise.cuda()

if args.sharing is not None :
    num_classes += -1
    if args.sepa_unknown_sharing :
        num_classes += -1


if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=range(1))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().cuda()
criterion2 = nn.BCELoss().cuda()
entropy = HLoss().cuda()
sharing_entropy = HLoss_for_3d_tensor().cuda()


# Training
def train(epoch):
    net.train()

    fake = netG(fixed_noise)
    torchvision.utils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(args.out_folder, epoch), normalize=True)

    train_loss = 0
    entropy_loss = 0
    correct = 0
    total = 0
    temp1_accum = 0
    temp2_accum = 0
    sigmoid_sum_loss = 0
    sharing_node_loss = 0
    num_classes = args.num_classes
    # optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), nesterov=True, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(args.lr, epoch), betas=(0.5,0.999), weight_decay=5e-4)
    if args.gan :
        # optimizerD = optim.SGD(netD.parameters(), lr=cf.learning_rate(1e-5, epoch), nesterov=True, momentum=0.9, weight_decay=5e-4)
        # optimizerG = optim.SGD(netG.parameters(), lr=cf.learning_rate(1e-3, epoch), nesterov=True, momentum=0.9, weight_decay=5e-4)
        optimizerD = optim.Adam(netD.parameters(), lr=cf.learning_rate(1e-6, epoch), betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=cf.learning_rate(2e-5, epoch), betas=(0.5, 0.999))
    

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        inputs, targets = Variable(inputs), Variable(targets)
        
        optimizer.zero_grad()
        outputs = net(inputs) 

        if args.gan :
            gan_target = torch.FloatTensor(targets.size()).fill_(0)
            uniform_dist = torch.Tensor(inputs.size(0), args.num_classes).fill_((1./args.num_classes))
            uniform_dist = Variable(uniform_dist)

            if use_cuda :
                gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()

            
            ###########################
            # (1) Update D network    #
            ###########################
            # train with real
            gan_target.fill_(real_label)
            targetv = Variable(gan_target)
            optimizerD.zero_grad()
            output = netD(inputs)
            errD_real = gan_criterion(output, targetv)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise = torch.FloatTensor(inputs.size(0), nz, 1, 1).normal_(0, 1).cuda()
            if use_cuda:
                noise = noise.cuda()
            noise = Variable(noise)
            fake = netG(noise)
            targetv = Variable(gan_target.fill_(fake_label))
            output = netD(fake.detach())
            errD_fake = 1.0 * gan_criterion(output, targetv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ###########################
            # (2) Update G network    #
            ###########################
            optimizerG.zero_grad()
            # Original GAN loss
            targetv = Variable(gan_target.fill_(real_label))  
            output = netD(fake)
            errG = 1.0 * gan_criterion(output, targetv)
            D_G_z2 = output.data.mean()

            # minimize the true distribution
            KL_fake_output = F.log_softmax(net(fake)[:,:num_classes])
            errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
            
            # targetv = Variable(gan_target.fill_(fake_label))
            # KL_fake_output = F.sigmoid(net(fake)[:,:num_classes])
            # errG_KL = criterion2(KL_fake_output, targetv)
            
            
            generator_loss = errG + 1.0 * errG_KL
            generator_loss.backward()
            optimizerG.step()

             # KL divergence
            noise = torch.FloatTensor(inputs.size(0), nz, 1, 1).normal_(0, 1).cuda()
            if use_cuda:
                noise = noise.cuda()
            noise = Variable(noise)
            fake = netG(noise)
            # KL_fake_output = F.sigmoid(net(fake)[:,:num_classes])
            # KL_loss_fake = 1.0 * criterion2(KL_fake_output, )*args.num_classes
            KL_fake_output = F.log_softmax(net(fake)[:,:num_classes])
            KL_loss_fake = 1.0 * F.kl_div(KL_fake_output, uniform_dist)*args.num_classes

            background_node = outputs[:,-1]

            fake_node_bce_loss = args.fake_node_bce_beta * criterion2(F.sigmoid(background_node), targetv)



                      # Forward Propagation

        num_sampling = num_classes
        if args.loss == 'ce' and args.sampling_rate != 1. :
            num_sampling = int(num_classes * args.sampling_rate)
            full_output = outputs
            outputs, targets = sampling_for_loss(outputs, targets, num_sampling, num_classes=num_classes, sharing=False)


        loss = criterion(outputs[:,:num_sampling], targets)  # Loss
        ce_loss = torch.zeros((1)).cuda()
        unknown_node_loss = torch.zeros((1)).cuda()
        temp1_accum = loss.detach().cpu() * (1./(batch_idx+1.)) + temp1_accum * (batch_idx/(batch_idx+1.))
        

        if args.loss == 'bce' :
            loss = torch.zeros(1).cuda()
        if 'bce' in args.loss :
            # if num_classes > 30 :
            #     num_sampling = int(num_classes * args.sampling_rate)
            #     outputs, targets = sampling_for_loss(outputs, targets, num_sampling)
            #     new_bce_targets = target_transform_for_elementwise_bce(targets, num_sampling)
            #     temp2 = args.bce_scale * criterion2(F.sigmoid(outputs[:,:num_sampling]), new_bce_targets).cuda()
            # else:
            num_sampling = int(num_classes * args.sampling_rate)
            if args.sampling_rate != 1. :
                full_output = outputs
                outputs, targets = sampling_for_loss(outputs, targets, num_sampling, num_classes=num_classes)
                new_bce_targets = target_transform_for_elementwise_bce(targets, num_sampling).cuda()
                temp2 = args.bce_scale * criterion2(F.sigmoid(outputs[:,:num_sampling]), new_bce_targets)
            else:
                if args.gan:
                # if False:
                    bce_targets = target_transform_for_elementwise_bce(targets, num_classes+1).cuda()
                    temp2 = args.bce_scale * criterion2(F.sigmoid(outputs), bce_targets).cuda()
                else:
                    bce_targets = target_transform_for_elementwise_bce(targets, num_classes).cuda()
                    temp2 = args.bce_scale * criterion2(F.sigmoid(outputs[:,:num_classes]), bce_targets).cuda()
                
            temp2_accum = temp2.detach().cpu() * (1./(batch_idx+1.)) + temp2_accum *  (batch_idx/(batch_idx+1.))
            
            if args.sigmoid_sum is not None :

                # sigmoid_sum = torch.sum(F.sigmoid(full_output[:,:num_classes]), dim=1)
                sigmoid_sum = torch.sum(F.sigmoid(full_output[:,:num_sampling]), dim=1)
                sigmoid_sum_loss = F.mse_loss(sigmoid_sum, args.sigmoid_sum * torch.ones_like(sigmoid_sum))
                loss += 0.5 * sigmoid_sum_loss
            
        

            loss += temp2

        if args.gan :
            loss += (KL_loss_fake + fake_node_bce_loss)

        entropy_loss = args.ent * entropy(outputs)


        if args.sharing is not None :
            classifictaion_target = Variable(torch.zeros(targets.size(0)).long()).cuda()
            output_target_select = outputs[:,:num_sampling].gather(dim=1, index=targets.unsqueeze(1))
            output_target_sharing_concat = torch.cat((output_target_select.view(-1,1), outputs[:,num_sampling].view(-1,1)),1)

            ce_loss = args.sharing * F.cross_entropy(F.softmax(output_target_sharing_concat), classifictaion_target)

            mask = torch.ones(outputs[:,:num_sampling].size()).byte().cuda()
            for i in range(mask.size(0)):
                mask[i, targets[i]] = torch.zeros(1)
                
            output_for_entropy_except_target_node = torch.masked_select(outputs[:,:num_sampling], mask)
            output_for_entropy_except_target_node = output_for_entropy_except_target_node.view(outputs[:,:num_sampling].size(0),-1)
            output_for_entropy_except_target_node = \
                torch.cat((output_for_entropy_except_target_node.view(targets.size(0),-1,1), outputs[:,num_sampling].view(-1,1,1).expand(targets.size(0),num_sampling-1,1)),2)
            # entropy_loss = args.ent * entropy(output_for_entropy_except_target_node)
            entropy_loss = args.ent * sharing_entropy(output_for_entropy_except_target_node)
            loss += ce_loss + entropy_loss

        # if args.ent != 0 and args.sharing is None:
        #     loss += entropy_loss
        
        train_loss += loss.item()
        max_logit, predicted = torch.max(outputs[:,:num_sampling].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if args.unknown_is_True != False:
            # weight = 0.01 * epoch
            weight = 0.1

            select1 = (F.sigmoid(max_logit) < 0.5)
            select2 = (F.sigmoid(outputs[:,targets]) < 0.1)
            select = (select1 + select2) >= 1
            if args.sepa_unknown_sharing :
                output_gather = F.sigmoid(outputs[:,num_classes+1].masked_select(select))
            else :
                output_gather = F.sigmoid(outputs[:,num_classes].masked_select(select))
            node_target = torch.ones(output_gather.size()).cuda()
            if output_gather.size(0) > 0 :
                
                # if weight >= 0.1 :
                #     weight = 0.1
                sharing_node_loss = weight * criterion2(output_gather, node_target)
                loss += sharing_node_loss

                if args.sepa_unknown_sharing :
                    select_unknown = (select1 + select2) < 1
                    unknown_output_gather = F.sigmoid(outputs[:,num_classes+1].masked_select(select_unknown))
                    if unknown_output_gather.size(0) > 0 :
                        unknown_node_target = torch.zeros(unknown_output_gather.size()).cuda()
                        unknown_node_loss = weight * criterion2(unknown_output_gather, unknown_node_target)
                        loss += unknown_node_loss
            else :
                pass
            # for i in range(outputs.size(0)):



        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        sys.stdout.write('\r')

        if args.gan:
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tD_x:%.2f D_G_z1:%.2f D_G_z2:%.2f BCE : %.4f Ent_los : %.4f Sha_los : %.4f node_loss : %.4f kl_fake : %.4f Acc@1: %.3f%%'
                        %(epoch, num_epochs, batch_idx+1,(len(trainset)//batch_size)+1, \
                            D_x, D_G_z1, D_G_z2,
                            temp2_accum, entropy_loss, ce_loss, fake_node_bce_loss.item(), KL_loss_fake.item(), float(100.00*float(correct)/float(total))))

        # if not args.sepa_unknown_sharing :
        #     if args.sigmoid_sum is not None :
        #         sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f CE_loss : %.4f BCE_loss : %.4f Ent_loss : %.4f Sha_loss : %.4f node_loss : %.4f sum_loss : %.4f Acc@1: %.3f%%'
        #                 %(epoch, num_epochs, batch_idx+1,
        #                     (len(trainset)//batch_size)+1, loss.item(), temp1_accum, temp2_accum, entropy_loss, ce_loss, sharing_node_loss, sigmoid_sum_loss, float(100.00*float(correct)/float(total))))
        #     else :
        #         sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f CE_loss : %.4f BCE_loss : %.4f Ent_loss : %.4f Sha_loss : %.4f node_loss : %.4f Acc@1: %.3f%%'
        #                 %(epoch, num_epochs, batch_idx+1,
        #                     (len(trainset)//batch_size)+1, loss.item(), temp1_accum, temp2_accum, entropy_loss, ce_loss, sharing_node_loss, float(100.00*float(correct)/float(total))))
        # else :
        #     sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f CE_loss : %.4f BCE_loss : %.4f Ent_loss : %.4f \
        #         Sha_loss : %.4f node_loss : %.4f unknown_loss : %.4f Acc@1: %.3f%%'
        #         %(epoch, num_epochs, batch_idx+1,(len(trainset)//batch_size)+1, 
        #             loss.item(), temp1_accum, temp2_accum, entropy_loss, ce_loss, \
        #                 sharing_node_loss, unknown_node_loss,float(100.00*float(correct)/float(total))))

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

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
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
        torch.save(state, save_point+file_name+'.t7')
        best_acc = acc
    if epoch > 100 and (epoch % 10 == 0 ):
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
