"""
A distributed mnist mlp using fsdp and fairscale.
This serves as a playground for experimentation with model and data parallel strategies similar to flexflow.
For a fully documented use case of fairscale, refer to multi_node_train.py
"""

# import libraries
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import time
import argparse


class Net(nn.Module):
    def __init__(self, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 10)

    def forward(self, x):
        # vanilla forward. also applies to model parallelism
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


def train_and_test(train_args):
    # =============================================================================
    # #Init
    # =============================================================================
    # print('initializing..')
    device = 'cuda:0'
    torch.cuda.set_device(device)

    dropout = 0.1
    n_workers = 1

    # optimizer
    optim_kwargs = {'lr': 2e-3, 'weight_decay': 1e-4, 'betas': (.9, .999)}

    # =============================================================================
    # Input
    # =============================================================================
    # 4) create dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    # create data loader
    kwargs = {'num_workers': n_workers, 'shuffle': False, 'drop_last': True, 'pin_memory': True,
              'batch_size': train_args.batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    # =============================================================================
    # Model
    # =============================================================================
    model = Net(dropout)
    model = model.to(device)
    # =============================================================================
    # Optimizer
    # =============================================================================
    loss_model = nn.CrossEntropyLoss()
    # for running without oss
    # optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    optimizer = optim.SGD(model.parameters(), 0.01)
    # =============================================================================
    # Train
    # =============================================================================
    model.train()
    start = time.time()
    avg_ips = 0
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(train_args.epochs):
            # if epoch == 1: start = time.time()
            start0 = time.time()
            train_loss = 0.0
            for data, target in data_loader:
                optimizer.zero_grad()

                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = loss_model(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(data_loader.dataset)
            cur_time = time.time() - start0
            ips = (len(data_loader.dataset)) / cur_time
            print(f'epochs : {epoch}, ips : {ips}')
            avg_ips += ips if epoch > 0 else 0

        tot_time = time.time() - start
        # ips = ((train_args.epochs-1) * train_args.batch_size * dist.get_world_size()) / tot_time
        avg_ips /= (train_args.epochs - 1)
        print(f"RANK = {device}, GPU AMOUNT = {1}, ELAPSED TIME = {tot_time}s,"
              f" AVG. THROUGHPUT = {avg_ips} samples/s")

        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        model.eval()    # prep model for *evaluation*

        for data, target in test_loader:
            target = target.to(device)
            data = data.to(device)
            output = model(data)
            loss = loss_model(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            for i in range(train_args.batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    # parser.add_argument("--print_params", action="store", default=False, type=bool)
    parser.add_argument("--print_params", action="store", default=True, type=bool)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    args = parser.parse_args()
    train_and_test(args)


