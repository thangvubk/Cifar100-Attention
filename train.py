from __future__ import print_function, division
from dataset import Cifar100_dataset
from torch.utils.data import DataLoader
from models import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import progressbar
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision
from resnet import *
from wide_resnet import *
import torch.backends.cudnn as cudnn

half = False
args={}
args['batch_size'] = 128
batch_size = 128
num_epochs = 200
def main():
    path = 'data/cifar-100-python/train'
    testpath = 'data/cifar-100-python/test'
    #trainset = Cifar100_dataset(path, sample_range=[0, 50000])
    #valset = Cifar100_dataset(testpath)#path, sample_range=[48000, 49000])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]) # meanstd transformation

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=args['batch_size'],
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=args['batch_size'],
                             shuffle=False, num_workers=4)
    model = resnet50()
    #model = ResNet(50,100)
    #model = Wide_ResNet(28, 10, 0.3, 100)
    print(model.parameters())
    print(sum([param.nelement() for param in model.parameters()]))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    #model.apply(conv_init)
    cudnn.benchmark = True
    if half:
        model.half()
    #optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        num_batches = len(trainset)//batch_size
        bar = progressbar.ProgressBar(max_value=num_batches)
        running_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            if half:
                inputs = inputs.half()
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.size(), labels.size())
            loss = loss_fn(outputs, labels)
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            bar.update(i, force=True)
        scheduler.step()
        print('Epoch %d: loss %f' %(epoch, running_loss/num_batches))

        print('Validating...')
        val_acc = 0
        num_batches = len(trainset)//batch_size + 1
        bar = progressbar.ProgressBar(max_value=num_batches)

        #running_loss = 0
        #for i, (inputs, labels) in enumerate(train_loader):
        #    inputs, labels = (Variable(inputs.cuda()),
        #                      Variable(labels.cuda()))
        #    outputs = model(inputs)
        #    loss = loss_fn(outputs, labels)
        #    running_loss += loss.data[0]
        #    outputs, labels = outputs.data, labels.data
        #    _, preds = outputs.topk(1, 1, True, True)
        #    preds = preds.t()
        #    corrects = preds.eq(labels.view(1, -1).expand_as(preds))
        #    val_acc += torch.sum(corrects)
        ##val_acc = val_acc.data[0]/len(valset)
        #print('train loss %f' %(running_loss/num_batches))
        #print('train acc', val_acc/len(trainset))

        print('Validating...')
        model.eval()
        val_acc = 0
        num_batches = len(valset)//batch_size + 1
        bar = progressbar.ProgressBar(max_value=num_batches)

        running_loss = 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            if half:
                inputs = inputs.half()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.data[0]
            outputs, labels = outputs.data, labels.data
            _, preds = outputs.topk(1, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(labels.view(1, -1).expand_as(preds))
            val_acc += torch.sum(corrects)
        #val_acc = val_acc.data[0]/len(valset)
        print('validation loss %f' %(running_loss/num_batches))
        print('Validation acc', val_acc/len(valset))
        print()

if __name__ == '__main__':
    main()
