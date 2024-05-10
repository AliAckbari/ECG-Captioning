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

def model_init(image_size: int, patch_size: int, num_classes: int, 
               dim: int, depth: int, heads: int,
               mlp_dim: int, dropout: float, emb_dropout: float) -> vit_pytorch.ViT:
    
    v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = dim,
    depth = depth,
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout
)
    
    return v
def train(model: torch.nn, trainloader: torch.utils.data.DataLoader, num_epochs: int = 10) -> None:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(perturbed_images)
            loss = criterion(outputs[1], labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 128 == 127:    # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Mini-batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

if "name" == "__main__":
    
    model = model_init()

