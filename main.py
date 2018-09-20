from __future__ import print_function, division
from torch.utils.data import DataLoader
from model import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
#import progressbar
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision
import torch.backends.cudnn as cudnn
import argparse
import os
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

args={}
parser = argparse.ArgumentParser()
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160],
                     help='Decrease learning rate at these epochs.')
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--att_type', type=str, default='no_attention')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--test-only', dest='test_only', action='store_true')
args = parser.parse_args()

def main():

    # Dataset
    print('Creating dataset...')
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, 
                np.array([63.0, 62.1, 66.7]) / 255.0)
            ]) 

    transform_val= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, 
                np.array([63.0, 62.1, 66.7]) / 255.0)
            ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    checkpoint = os.path.join(args.checkpoint, args.model)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    model_path = os.path.join(checkpoint, 'best_model.pt')

    print('Loading model...')
    opt = {'name': args.model,
           'att_type': args.att_type}

    model = get_model(opt)

    if args.test_only:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            raise Exception('Cannot find model', model_path)
    print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    # tensor board
    tb = SummaryWriter(checkpoint)
    
    # Test only
    if args.test_only:
        print('Testing...')
        model.eval()
        acc = 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            outputs = model(inputs)
            outputs, labels = outputs.data, labels.data
            _, preds = outputs.topk(1, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(labels.view(1, -1).expand_as(preds))
            acc += torch.sum(corrects)
        acc = acc.item()/len(valset)*100
        print('Accuracy: %.2f' %acc)
        return

    # optim
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.2)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = -1
    # Train and val
    for epoch in range(args.num_epochs):
        # Train
        learning_rate = optimizer.param_groups[0]['lr']
        print('Start training epoch {}. Learning rate {}'.format(epoch, learning_rate))
        model.train()
        num_batches = len(trainset)//args.batch_size
        running_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.data.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = running_loss/num_batches
        print('Training loss %f' %train_loss)

        # Validate
        model.eval()
        val_acc = 0
        num_batches = len(valset)//args.batch_size + 1
        running_loss = 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.data.item()
            outputs, labels = outputs.data, labels.data
            _, preds = outputs.topk(1, 1, True, True)
            preds = preds.t()
            corrects = preds.eq(labels.view(1, -1).expand_as(preds))
            val_acc += torch.sum(corrects)
        val_acc = val_acc.item()/len(valset)*100
        val_loss = running_loss/num_batches
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        print('Validation loss %f' %(running_loss/num_batches))
        print('Validation acc', val_acc)
        print()

        #update tensorboard
        tb.add_scalar('Learning rate', learning_rate, epoch)
        tb.add_scalar('Train loss', train_loss, epoch)
        tb.add_scalar('Val loss', val_loss, epoch)
        tb.add_scalar('Val acc', val_acc, epoch)

    print('Best validation acc %.2f' %best_val_acc)

if __name__ == '__main__':
    main()
