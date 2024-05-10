import torch
from vit_pytorch import ViT
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

def prepare_test_train() -> tuple:
    transform = transforms.Compose([transforms.ToTensor()])


    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)

    batch_size = 50


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return trainloader, testloader, device




if "name" == "__main__":
    


