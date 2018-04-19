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


args={}
args['batch_size'] = 256
batch_size = 256
num_epochs = 100
def main():
    path = 'data/cifar-100-python/train'
    trainset = Cifar100_dataset(path, sample_range=[0, 49000])
    valset = Cifar100_dataset(path, sample_range=[49000, 50000])
    train_loader = DataLoader(trainset, batch_size=args['batch_size'],
                             shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args['batch_size'],
                             shuffle=False, num_workers=4)

    model = resnet50()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        num_batches = len(trainset)//batch_size
        bar = progressbar.ProgressBar(max_value=num_batches)
        running_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.size(), labels.size())
            loss = loss_fn(outputs, labels)
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            bar.update(i+1, force=True)
        print('epoch %d: loss %f' %(epoch, running_loss/num_batches))

        print('Validating...')
        val_acc = 0
        num_batches = len(valset)//batch_size + 1
        bar = progressbar.ProgressBar(max_value=num_batches)
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            outputs = model(inputs)
            _, preds = outputs.max(1)
            labels = labels.squeeze()
            val_acc += sum(preds == labels)
            #bar.update(i+1)
        val_acc = val_acc.data[0]/len(valset)
        print('Validation acc %.2f' %val_acc)
        print()

if __name__ == '__main__':
    main()
