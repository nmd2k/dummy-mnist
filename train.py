import argparse

from utils.model import Net
from utils.utils import get_dataloader, get_dataset, get_transform

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--cuda', action='store_true', default=False)
    return parser.parse_args()


def train(model, epoch, trainloader, optimizer, loss_function, cuda=None):
    model.train()
    running_loss = 0
    for i, (input, target) in enumerate(trainloader, 0):
        if cuda:
            input, target = input.cuda(), target.cuda()
        predict = model(input)
        loss = loss_function(predict, target)
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss/len(trainloader.dataset)
    print(f'Epoch {epoch+1} | Train: Loss=[{total_loss:.2f}]')
    return total_loss


def test(model, epoch, testloader, loss_function, cuda=None):
    model.eval()
    test_loss = 0
    count_correct   = 0
    with torch.no_grad():
        for idx, (input, target) in enumerate(testloader):
            if cuda:
                input, target = input.cuda(), target.cuda()
            predict = model(input)
            loss = loss_function(predict, target)
            test_loss += loss.item()
            # print(predict.shape, target.shape)

            count_correct += torch.sum(torch.argmax(predict, dim=1) == target).item()
    
    test_loss /= len(testloader)
    test_accuracy = count_correct / len(testloader.dataset)
    print(f'Epoch {epoch+1} | Test:  Loss=[{test_loss:.2f}]; Acc=[{test_accuracy:.2f}]')
    return test_loss, test_accuracy


if __name__ == '__main__':
    opt = args_parser()
    epochs, lr, momentum, dropout_rate, \
    batch_size, device = opt.epoch, opt.lr, opt.momentum, \
                         opt.dropout_rate, opt.bs, opt.cuda
    if opt.cuda:
        torch.cuda.set_device(0)
        device = torch.device('cuda')

    train_set, test_set = get_dataset(transform=get_transform())
    trainloader, testloader = get_dataloader(train_set, test_set, batch_size)

    # init model
    model = Net(dropout_rate)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer     = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # training
    print(f'[INFO] Training {model.__class__.__name__} {epochs} epochs...')
    train_losses, test_losses, test_accuracy = [], [], []

    for epoch in range(epochs):
        train_loss = train(model, epoch, trainloader, optimizer, loss_function, device)
        train_losses.append(train_loss)

        test_loss, test_acc = test(model, epoch, testloader, loss_function, device)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
