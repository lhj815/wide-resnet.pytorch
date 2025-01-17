############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
num_epochs = 300
batch_size = 64
optim_type = 'SGD'

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'SVHN': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'SVHN': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if num_epochs == 99 :
        if(epoch > 70):
            optim_factor = 5
        elif(epoch > 50):
            optim_factor = 3
        elif(epoch > 30):
            optim_factor = 2
        elif(epoch > 5):
            optim_factor = 1
        return init*math.pow(0.2, optim_factor)
    if num_epochs == 103 :
        if(epoch > 100):
            optim_factor = 5
        elif(epoch > 90):
            optim_factor = 3
        elif(epoch > 70):
            optim_factor = 2
        elif(epoch > 50):
            optim_factor = 1
        return init*math.pow(0.2, optim_factor)

    if num_epochs < 150:
        if(epoch > 100):
            optim_factor = 6
        elif(epoch > 90):
            optim_factor = 5
        elif(epoch > 80):
            optim_factor = 3
        elif(epoch > 60):
            optim_factor = 2
        elif(epoch > 30):
            optim_factor = 1
        return init*math.pow(0.2, optim_factor)
    
    # if(epoch > 190):
    #     optim_factor = 6
    if(epoch > 190):
        optim_factor = 5
    elif(epoch > 170):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1


    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
