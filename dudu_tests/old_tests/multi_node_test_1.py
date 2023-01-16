import os
import time
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from fairscale.optim.grad_scaler import ShardedGradScaler


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
# WORLD_SIZE = 8
# EPOCHS = 2

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # make sure all weights are deterministic
        torch.manual_seed(0)
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 200),
            nn.ReLU(),
            nn.Linear(200, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.convnet(x)


def init_processes(epochs):
    # These are the parameters used to initialize the process group
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print("rank=", rank)
    print("local_rank=", local_rank, flush=True)
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    # dist.init_process_group(backend='nccl', init_method="tcp://localhost:29502", rank=local_rank, world_size=local_world_size)
    # currently its assumed that it will use the default initialization method to sync all workers across nodes
    dist.init_process_group(backend='nccl')
    train(local_rank, world_size, epochs)
    dist.destroy_process_group()


def train(
    rank: int,
    world_size: int,
    epochs: int):

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}"
    )
    # Create distributed sampler pinned to rank
    sampler = DistributedSampler(trainset,
                                 num_replicas=world_size,
                                 rank=rank,
                                 shuffle=True,  # May be True
                                 seed=42)

    # Wrap train dataset into DataLoader
    trainloader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,  # Must be False!
                              num_workers=0,
                              sampler=sampler,
                              pin_memory=True)
    print(f"dataloader len: {len(trainloader)}, for rank: {rank}")

    # # Create distributed sampler pinned to rank
    # testsampler = DistributedSampler(testset,
    #                              num_replicas=world_size,
    #                              rank=rank,
    #                              shuffle=True,  # May be True
    #                              seed=42)

    # Wrap train dataset into DataLoader
    testloader = DataLoader(testset,
                              batch_size=batch_size,
                              shuffle=False,  # Must be False!
                              num_workers=0,
                              # sampler=testsampler,
                              pin_memory=True)
    print(f"testloader len: {len(testloader)}, for rank: {rank}")

    torch.cuda.set_device(rank)
    # Problem statement
    net = Net()
    # if os.path.exists("state_G.ckpt"):
    #     dist.barrier()
    #     net.load_state_dict(torch.load("state_G.ckpt"))
    net.to(rank)
    # TODO print the network weights at the end foreach rank to make sure its identical
    criterion = nn.CrossEntropyLoss()

    # Wrap the optimizer in its state sharding brethren
    # optimizer = OSS(params=net.parameters(), optim=base_optimizer, **base_optimizer_arguments)

    # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
    # they say its important to define the the sharded optimizer before the sharded network
    # model = ShardedDDP(net, optimizer)

    # Wrap the model into FSDP, which will reduce parameters to the proper ranks
    # model = FSDP(model)
    model = FSDP(net, verbose=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    # testing train times
    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            model.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = batch[0].cuda(rank), batch[1].cuda(rank)
            inputs, labels = batch[0].to(rank), batch[1].to(rank)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0 and rank == 0:    # print every 100 mini-batches on rank 0
                print('[epoch %d, batch %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 100))
                running_loss = 0.0

    tot_time = time.time() - start
    ips = (epochs * len(trainloader) * world_size) / tot_time
    print(f"GPU AMOUNT = {world_size}, ELAPSED TIME = {tot_time}s, THROUGHPUT = {ips} samples/s")
    print(f'rank {rank} Finished Training')

    # model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    for batch in testloader:
        model.zero_grad()
        inputs, labels = batch[0].to(rank), batch[1].to(rank)
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--epochs", action="store", default=2, type=int)
    args = parser.parse_args()
    init_processes( args.epochs)
