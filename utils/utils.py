from numpy import save
import torch
import torchvision
import torchvision.transforms as transforms


def get_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    return transform


def get_dataset(transform, download=True, save_path='./data'):
    train_set = torchvision.datasets.MNIST(root=save_path, train=True, download=download, transform=transform)
    test_set  = torchvision.datasets.MNIST(root=save_path, train=False, download=download, transform=transform)
    return train_set, test_set


def get_dataloader(train_set, test_set, batch_size):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
