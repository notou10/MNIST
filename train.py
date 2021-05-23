import os
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torchvision import datasets, transforms
import numpy as np
from torch.nn.modules.activation import LeakyReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#1 hyperparameter, ckpt_dir 생성
num_epoch = 18
batch_size = 64
lr = 1e-3

ckpt_dir = './ckpt_dir'
log_dir = './log_dir'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#classifier network 구축
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, stride = 1, padding = 0, kernel_size=5, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, stride = 1, padding=0, kernel_size=5, bias = True)
        self.drop2 = nn.Dropout2d(0.5)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        
        self.fcn1 = nn.Linear(in_features=320, out_features=50, bias = True)
        self.fcn1_relu = nn.ReLU()
        self.fcn1_dropout = nn.Dropout2d(0.5)

        self.fcn2 = nn.Linear(in_features=50, out_features=10, bias = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 320)
        x = self.fcn1(x)
        x = self.fcn1_relu(x)
        x = self.fcn1_dropout(x)

        x = self.fcn2(x)

        return x

#3 save, load 함수 구축
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim' : optim.state_dict()}, './%s/epoch%s.pth'%(ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    loaded_model = torch.load('./%s/%s'%(ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(loaded_model['net'])
    optim.load_state_dict(loaded_model['optim'])

    return net, optim

#4 data 불러오기
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, ), std = (0.5, ))]) #0~1 -> -1~1
dataset = datasets.MNIST(root = './', train = True, transform = transform, download = True)
loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

num_data = len(loader.dataset)
num_batch = np.ceil(num_data/batch_size)

#5 loss 구하기
net = Net().to(device)
param = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim = 1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params = param, lr = lr)
writer = SummaryWriter(log_dir = log_dir)

#6 network training
for epoch in range(1, num_epoch):
    net.train()

    loss_arr = []
    acc_arr = []
    
    for batch_idx, (input, label) in enumerate(loader, 1):
        input = input.to(device)
        label = label.to(device)

        output = net(input)
        pred = fn_pred(output)

        optim.zero_grad()

        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)

        loss.backward()
        optim.step()

        loss_arr += [loss.item()] #loss를 tensor가 아닌 scalar값으로
        acc_arr += [acc.item()]

        print('EPOCH : %s | BATCH : %04d/%04d | LOSS : %.4f | ACC : %.4f'%(epoch, batch_idx, num_batch, np.mean(loss_arr), np.mean(acc_arr)))


    save(ckpt_dir = ckpt_dir, net = net, optim = optim, epoch = epoch)
