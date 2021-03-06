import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets', flatten=False):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    if flatten:
        trans.append(torchvision.transforms.Lambda(lambda x: torch.flatten(x)))
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def load_data_cifar10(batch_size, resize=None, root='~/Datasets', flatten=False):
    "Download the cifar 10 dataset and load to memory, origin size per pic (32,32,3)"
    trans = []
    trans.append(torchvision.transforms.ToTensor())
    trans.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(trans)

    cifar10_train = torchvision.datasets.CIFAR10(root=root, train=True,download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def load_data_cifar100(batch_size, resize=None, root='~/Datasets', flatten=False):
    "Download the cifar100 dataset and load to memory, origin size per pic (32,32,3)"
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    if flatten:
        trans.append(torchvision.transforms.Lambda(lambda x: torch.flatten(x)))
    transform = torchvision.transforms.Compose(trans)

    cifar100_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # ???????????????device?????????net???device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # ????????????, ????????????dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # ??????????????????
            else:  # ??????????????????, 3.13?????????????????????, ?????????GPU
                if ('is_training' in net.__code__.co_varnames):  # ?????????is_training????????????
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


def visualize_incumbent(data):
    writer = SummaryWriter(write_to_disk=True)
    n_iter = 0
    for arr in data:
        n_iter += 1
        arr.sort()
        writer.add_scalar('data/best', arr[-1], n_iter)
    writer.close()


class DeepFCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.add_module('first_relu', nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc3 = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()) for i in range(3)
            ]
        )
        self.fc4 = nn.ModuleDict({
            'fc4-1': nn.Linear(64, 32),
            'relu': nn.ReLU()
        })
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.first_relu(x)
        x = self.fc2(x)
        for i, l in enumerate(self.fc3):
            x = l(x)
        x = self.fc4['fc4-1'](x)
        x = self.fc4['relu'](x)
        y_hat = self.fc5(x)
        return y_hat


def fast_train_dense(net, num_epochs):
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size,
                                                       resize=15, flatten=True)  # ????????????out of memory??????????????????????????????batch_size???resize
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    res = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    net.performance = res['incumbent_test_accuracy']


class AlexNetMnist(nn.Module):
    def __init__(self):
        super(AlexNetMnist, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # ????????????????????????????????????2?????????????????????????????????????????????????????????????????????
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # ??????3??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            # ??????????????????????????????????????????????????????????????????
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # ????????????????????????????????????LeNet???????????????????????????????????????????????????
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # ??????????????????????????????Fashion-MNIST????????????????????????10?????????????????????1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def fast_train_alex(net, num_epochs):
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224, flatten=False)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    res = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    net.performance = res['incumbent_test_accuracy']

def fast_evaluate_alex(net):
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224, flatten=False)
    res = evaluate_accuracy(test_iter, net)
    net.performance = res
    return res

def shirt_evaluate_alex(net):
    alex = net
    train_iter, test_iter = load_data_fashion_mnist(batch_size=128, resize=224, flatten=False)
    classes = ('t-shirt', 'trouser', 'pullover', 'dress',
            'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')
    # prepare to count Recall for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # Prepare to count Precision for each class
    TP = {classname: 0 for classname in classes}
    TPFP = {classname: 0 for classname in classes}
    # again no gradients needed
    # ?????????: 
    # ?????????: precision : TP/(TP+FP)
    with torch.no_grad():
        for data in test_iter:
            images, labels = data
            outputs = alex(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    TP[classes[prediction]] += 1
                total_pred[classes[label]] += 1
                TPFP[classes[prediction]] += 1
    for classname in classes:
        if classname == 'shirt':
            recall = 100 * float(correct_pred[classname]) / total_pred[classname]
            precision = 100 * float(TP[classname]) / TPFP[classname]
            print(f'For class: {classname:5s}, Recall is {recall:.1f} % , Precision is {precision:.1f} %')
            net.performance = precision
            return precision


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

def fast_train_le(net, num_epochs):
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=28, flatten=False)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    res = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    net.performance = res['incumbent_test_accuracy']

def fast_evaluate_le(net):
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=28, flatten=False)
    res = evaluate_accuracy(test_iter, net)
    net.performance = res
    return res


class AlexNetCifar(nn.Module):
    def __init__(self):
        super(AlexNetMnist, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # ????????????????????????????????????2?????????????????????????????????????????????????????????????????????
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # ??????3??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            # ??????????????????????????????????????????????????????????????????
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # ????????????????????????????????????LeNet???????????????????????????????????????????????????
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # ??????????????????????????????Fashion-MNIST????????????????????????10?????????????????????1000
            nn.Linear(4096, 100),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def fast_train_alex_cifar100(net, num_epochs):
    batch_size = 128
    train_iter, test_iter = load_data_cifar100(batch_size, resize=224, flatten=False)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    res = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    net.performance = res['incumbent_test_accuracy']

def fast_evaluate_alex_cifar100(net):
    batch_size = 128
    train_iter, test_iter = load_data_cifar100(batch_size, resize=224, flatten=False)
    res = evaluate_accuracy(test_iter, net)
    net.performance = res
    return res


# ResNet part
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = FlattenLayer()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.flatten_layer(output)
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


def fast_train_resnet18_cifar100(net, num_epochs):
    batch_size = 128
    train_iter, test_iter = load_data_cifar100(batch_size, resize=224, flatten=False)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    res = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    net.performance = res['incumbent_test_accuracy']

def fast_evaluate_resnet18_cifar100(net):
    batch_size = 128
    train_iter, test_iter = load_data_cifar100(batch_size, resize=224, flatten=False)
    res = evaluate_accuracy(test_iter, net)
    net.performance = res
    return res