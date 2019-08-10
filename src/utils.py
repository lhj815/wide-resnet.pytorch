import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np




def draw_graph(dir_name, acc):
    cifar = np.loadtxt('%s/confidence_Base_In.txt'%dir_name, delimiter=',')
    other = np.loadtxt('%s/confidence_Base_Out.txt'%dir_name, delimiter=',')
    Y1 = other
    X1 = cifar
    bins = np.arange(0,1.01, step=0.02)
    plt.hist(X1, bins=bins, alpha=0.3, color='black', label='in', rwidth= 0.7, weights=np.ones(len(X1)) / len(X1))
    plt.hist(Y1, bins=bins, alpha=0.3, color='orange', label='out', rwidth=0.7, weights=np.ones(len(Y1)) / len(Y1))
    plt.ylim([0,1])
    plt.axvline(X1.mean(), color='black', linestyle='dashed', linewidth=1)
    plt.axvline(Y1.mean(), color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(acc/100., color='red', linestyle='dashed', linewidth=2, label='acc:{:.1f}%'.format(acc))
    plt.legend(loc='upper right')
    plt.savefig('{}/graph_{}acc.png'.format(dir_name,acc))


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = 1.0 * b.sum(dim=1)          # not -1.0 -> for maximize entropy
        b = b.mean()                    # batch_wise mean
        return b

class HLoss_for_3d_tensor(nn.Module):
    def __init__(self):
        super(HLoss_for_3d_tensor, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
        b = 1.0 * b.sum(dim=2)          # not -1.0 -> for maximize entropy
        b = b.mean()                    # batch_wise mean
        return b

def target_transform_for_elementwise_bce(targets, num_class=10, bias=0):
    one_hot_target = torch.zeros((targets.size(0), num_class)).cuda()
    
    for i in range(len(targets)) :
        one_hot_target[i][targets[i]] = 1
    if bias != 0 :
        one_hot_target[:,num_class-1] = bias
    return one_hot_target.detach()


def target_soft_transform_for_elementwise_bce(targets, num_class=10, bias=0):
    
    one_hot_target = torch.zeros((targets.size(0), num_class))
    multi_gau = torch.distributions.multivariate_normal.MultivariateNormal(one_hot_target, torch.eye(num_class)/100.0)
    single_gau = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.01]))
    noise = multi_gau.sample()
    one_hot_target += torch.abs(noise)
    
    for i in range(len(targets)) :
        single_noise = single_gau.sample()
        one_hot_target[i][targets[i]] = torch.tensor([1]).float() - torch.abs(single_noise)
    if bias != 0 :
        one_hot_target[:,num_class-1] = bias
    
    return one_hot_target.cuda().detach()

def write_output(output, target, output_folder, epoch, num_classes=10, mode='ce', cifar100=False):
    f1 = open('{}/test_confidence_{}epoch_right.txt'.format(output_folder, epoch), 'a')
    f2 = open('{}/test_confidence_{}epoch_wrong.txt'.format(output_folder, epoch), 'a')
    f3 = open('{}/test_confidence_{}epoch_right_all_class.txt'.format(output_folder, epoch), 'a')
    f4 = open('{}/test_confidence_{}epoch_wrong_classwise.txt'.format(output_folder, epoch), 'a')
    if mode == 'ce' :
        output = F.softmax(output)
    elif mode == 'bce':
        output = F.sigmoid(output)
    else:
        print('loss_error')
        exit()

    max_output, pred = output[:,:num_classes].data.max(1) # get the index of the max log-probability
    min_output, min_pred = output[:,:num_classes].data.min(1)
    
    # if len(output[0]) > 11 and mode == 'bce':
    #     target = torch.zeros(pred.size()).cuda().long()
    for i in range(len(pred)):
        if cifar100 :
            f4.write("(cifar100-->{})  (cifar10-->{}){:.3f}\n".format(  target[i].item(), pred[i].item(), max_output[i].item() )   )
            sorted, indices = torch.sort(output[i,:num_classes].clone(), descending=True)
            sorted = sorted.cpu().data.detach().numpy()
            f4.write("({}){:.3f}  ({}){:.3f}  ({}){:.3f}  ({}){:.3f}  ({}){:.3f}  (-1){:.3f}\n"
                .format(indices[0],sorted[0], indices[1], sorted[1], indices[2], sorted[2], indices[3], sorted[3], indices[4], sorted[4] , output[i,-1]))
            continue
        if pred[i] == target[i]:
            if len(output[i]) < 11 :
                f1.write("{}\t{}\t{}\t\t\t{:.3f}\t{:.3f}\t\n".format(target[i], pred[i], min_pred[i], max_output[i],  min_output[i]))
                f3.write("{:.3f}  {:.3f}\n{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}".format(
                    output[i,target[i]].item(),max_output[i].item(),
                    output[i,0].item(),output[i,1].item(),output[i,2].item(),output[i,3].item(), output[i,4].item(),
                    output[i,5].item(),output[i,6].item(),output[i,7].item(),output[i,8].item(), output[i,9].item()
                    ))
                if len(output[i]) == 11 :
                    f3.write("  {:.3f}".format(output[i,-1].item()))
                f3.write("\n")
                # f3.write("{:.3f}  {:.3f}\n{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}\n".format(
                #     output[i,target[i]].item(),max_output[i].item(),
                #     output[i,0].item(),output[i,1].item(),output[i,2].item(),output[i,3].item(), output[i,4].item(),
                #     output[i,5].item(),output[i,6].item(),output[i,7].item(),output[i,8].item(), output[i,9].item(), output[i,10].item()
                #     ))
            else :
                sorted, indices = torch.sort(output[i,:num_classes].clone(), descending=True)
                sorted = sorted.cpu().data.detach().numpy()
                f1.write("{}\t{}\t{}\t\t\t{:.3f}\t{:.3f}\t\n".format(target[i], pred[i], min_pred[i], max_output[i],  min_output[i]))
                f3.write("({}){:.3f}  ({}){:.3f}\n".format(  target[i].item(), output[i,target[i]].item(), pred[i].item(), max_output[i].item() )   )
                f3.write("({}){:.3f}  ({}){:.3f}  ({}){:.3f}  ({}){:.3f}  ({}){:.3f}  (-1){:.3f}\n"
                    .format(indices[0],sorted[0], indices[1], sorted[1], indices[2], sorted[2], indices[3], sorted[3], indices[4], sorted[4], output[i,-1]))
            
        else :
            if len(output[i]) < 11 :
                # f2.write("{}\t{}\t{}\t\t\t{:.3f}\t{:.3f}\t{:.3f}\t\n".format(target[i], pred[i], min_pred[i], output[i,target[i]].item(), max_output[i],  min_output[i]))
                f4.write("{:.3f}  {:.3f}\n{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}".format(
                    output[i,target[i]].item(),max_output[i].item(),
                    output[i,0].item(),output[i,1].item(),output[i,2].item(),output[i,3].item(), output[i,4].item(),
                    output[i,5].item(),output[i,6].item(),output[i,7].item(),output[i,8].item(), output[i,9].item()
                    ))
                if len(output[i]) == 11 :
                    f4.write("  {:.3f}".format(output[i,-1].item()))
                f4.write("\n")
                # f4.write("{:.3f}  {:.3f}\n{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}\n".format(
                #     output[i,target[i]].item(),max_output[i].item(),
                #     output[i,0].item(),output[i,1].item(),output[i,2].item(),output[i,3].item(), output[i,4].item(),
                #     output[i,5].item(),output[i,6].item(),output[i,7].item(),output[i,8].item(), output[i,9].item(), output[i,10].item()
                #     ))
            else :
                f4.write("({}){:.3f}  ({}){:.3f}\n".format(  target[i].item(), output[i,target[i]].item(), pred[i].item(), max_output[i].item() )   )
                sorted, indices = torch.sort(output[i,:num_classes].clone(), descending=True)
                sorted = sorted.cpu().data.detach().numpy()
                f4.write("({}){:.3f}  ({}){:.3f}  ({}){:.3f}  ({}){:.3f}  ({}){:.3f}  (-1){:.3f}\n"
                    .format(indices[0],sorted[0], indices[1], sorted[1], indices[2], sorted[2], indices[3], sorted[3], indices[4], sorted[4] , output[i,-1]))
                # sort = sorted(output[i].clone().cpu().data.detach().numpy(), reverse=True)
                # f4.write("{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}\n".format(sort[0], sort[1], sort[2], sort[3], sort[4]))

    f1.close()
    f2.close()
    f3.close()
    f4.close()


def count(pred, num_classes, data='out'):
    if data == 'out':
        bag = torch.zeros((num_classes)).cuda()
        for i in range(len(pred)):
            bag[pred[i]] += 1
    else :
        print()

    return bag

def sampling_for_loss(outputs=None, targets=None, num_of_sampling=10, num_classes=100, sharing=True):
    # target : [batch_size, number of class]
    batch_size = outputs.size(0)
    # uniform_distrib = distrib.uniform.Uniform(torch.tensor([0]), torch.tensor([num_class-1]))
    # sampled_uniform = uniform_distrib.sample(torch.Size([batch_size, 10])))
    list = (range(num_classes))
    sampled_list = [torch.tensor((random.sample(list, num_of_sampling))).cuda() for i in range(batch_size)]
    exist = [targets[i].long() in sampled_list[i] for i in range(len(sampled_list))]
    changer = [targets[i] if (exist[i] is False) else sampled_list[i][0] for i in range(len(sampled_list))]

    for i in range(batch_size):
        if exist[i]:
            swap_idx = np.where(sampled_list[i].cpu().numpy() == targets[i].item())
            tempppp = sampled_list[i][swap_idx].detach()
            sampled_list[i][swap_idx] = sampled_list[i][0]
            sampled_list[i][0] = tempppp
        sampled_list[i][0] = changer[i]
        if sharing :
            temp = torch.zeros(num_of_sampling+1).long().cuda()
            temp[:num_of_sampling] = sampled_list[i]
            temp[num_of_sampling] = num_classes
            sampled_list[i] = temp
    if sharing :
        num_of_sampling += 1

    new_index = sampled_list


    new_outputs = torch.zeros((batch_size, num_of_sampling)).cuda()
    new_targets = torch.zeros((batch_size)).long().cuda()
    
    for i in range(batch_size):
        new_outputs[i] = outputs[i][new_index[i][:]]
    # print('output[0]:', outputs[0], '| target[0] :', targets[0])
    # print('new_index[0] :', new_index[0])

    return new_outputs, new_targets


def sampling_for_cifar100(outputs=None, targets=None, num_of_sampling=10, num_classes=100, sharing=False):
    # target : [batch_size, number of class]
    batch_size = outputs.size(0)
    # uniform_distrib = distrib.uniform.Uniform(torch.tensor([0]), torch.tensor([num_class-1]))
    # sampled_uniform = uniform_distrib.sample(torch.Size([batch_size, 10])))
    list = (range(num_classes))
    sampled_list = [torch.tensor((random.sample(list, num_of_sampling))).cuda() for i in range(batch_size)]
    exist = [targets[i].long() in sampled_list[i] for i in range(len(sampled_list))]
    changer = [targets[i] if (exist[i] is False) else sampled_list[i][0] for i in range(len(sampled_list))]

    for i in range(batch_size):
        if exist[i]:
            swap_idx = np.where(sampled_list[i].cpu().numpy() == targets[i].item())
            tempppp = sampled_list[i][swap_idx].detach()
            sampled_list[i][swap_idx] = sampled_list[i][0]
            sampled_list[i][0] = tempppp
        sampled_list[i][0] = changer[i]
        if sharing :
            temp = torch.zeros(num_of_sampling+1).long().cuda()
            temp[:num_of_sampling] = sampled_list[i]
            temp[num_of_sampling] = num_classes
            sampled_list[i] = temp
    if sharing :
        num_of_sampling += 1

    new_index = sampled_list


    new_outputs = torch.zeros((batch_size, num_of_sampling+9)).cuda()
    new_targets = torch.zeros((batch_size, num_of_sampling+9)).float().cuda()
    new_targets[:,:10] = 1
    for i in range(batch_size):
        new_outputs[i,9:] = outputs[i][new_index[i][:]]
        new_outputs[i,:9] = new_outputs[i,9]
    # print('output[0]:', outputs[0], '| target[0] :', targets[0])
    # print('new_index[0] :', new_index[0])
    return new_outputs, new_targets