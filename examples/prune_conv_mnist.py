import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import time, math
from copy import deepcopy

import torch_pruning as tp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    # trans.append(torchvision.transforms.Lambda(lambda x: torch.flatten(x)))
    transform = torchvision.transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # ???????????????device?????????net???device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # ????????????, ????????????dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # ??????????????????
            else: # ??????????????????, 3.13?????????????????????, ?????????GPU
                if('is_training' in net.__code__.co_varnames): # ?????????is_training????????????
                    # ???is_training?????????False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    incumbent_test_accuracy = 0
    incumbent_epoch = 0
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        if incumbent_test_accuracy < test_acc:
            incumbent_test_accuracy = test_acc
            incumbent_epoch = epoch
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return {'incumbent_epoch': incumbent_epoch, 'incumbent_test_accuracy': incumbent_test_accuracy}


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 1, 28, 28))
    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune_conv(m, 0.2)
    return model


def exp():
    result = []
    net = LeNet()
    net_orig = deepcopy(net)
    batch_size = 128

    print("------------------------------pruned_model_from_scratch------------------------------")
    # ???????????????
    idxs = {}
    strategy = tp.strategy.L1Strategy()
    static_layer = ['4']
    for i, layer in net.fc.named_children():
        if i not in static_layer:
            if isinstance(layer, nn.Linear):
                idxs[i] = strategy(layer.weight, amount = 0.4)
    DG = tp.DependencyGraph()
    DG.build_dependency(net.fc, example_inputs=torch.randn(1,16*4*4))

    pruning_plans = []
    for name, module in net.fc.named_children():
        if name in idxs:
            pruning_plans.append(DG.get_pruning_plan(module, tp.prune_linear, idxs=idxs[name]))

    print(pruning_plans)

    print(net)

    for plan in pruning_plans:
        plan.exec()
    print(net)

    prune_model(net)
    print(net)

    # train from scratch
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    lr, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    res = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    print(res)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^pruned_model_from_scratch_ends^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    return res

if __name__ == '__main__':
    exp()
