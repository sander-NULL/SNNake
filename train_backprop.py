'''
Created on Apr 18, 2024

@author: sander
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import itertools
import numpy as np
import snake_core as sc

def translate(string):
    if string == "LEFT":
        return torch.tensor([1.0,0,0,0])
    elif string == "UP":
        return torch.tensor([0,1.0,0,0])
    elif string == "RIGHT":
        return torch.tensor([0,0,1.0,0])
    elif string == "DOWN":
        return torch.tensor([0,0,0,1.0])

def normalize(inputvec):
    out = inputvec
    out[0] = 2/sc.FIELD_WIDTH * inputvec[0] - 1   #maps it to [-1, 1)
    out[1] = 1 - 2/sc.FIELD_HEIGHT * inputvec[1]  #maps it to (-1, 1]
    out[4] = 2/sc.FIELD_WIDTH * inputvec[4] - 1   #maps it to [-1, 1)
    out[5] = 1 - 2/sc.FIELD_HEIGHT * inputvec[5]  #maps it to (-1, 1]
    return out

class CustomTrainSet(Dataset):
    def __init__(self, annotations_file, transform = None, target_transform = None):
        self.annotations = open(annotations_file, newline='')
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        self.annotations.seek(0)
        return sum(1 for _ in self.annotations)
    
    def __getitem__(self, idx):
        self.annotations.seek(0)
        reader = csv.reader(self.annotations)
        line = next(itertools.islice(reader, idx, None))
        head_x = float(line[0])
        head_y = float(line[1])
        head_x_change = float(line[2])
        head_y_change = float(line[3])
        food_x = float(line[4])
        food_y = float(line[5])
        inputvec = np.array([head_x, head_y, head_x_change, head_y_change, food_x, food_y])
        return self.transform(inputvec), self.target_transform(line[6])

class NormedSigmoid(nn.Module):
    def __init__(self):
        super(NormedSigmoid, self).__init__()
    
    def forward(self, x):
        sigm = nn.Sigmoid()
        ret = nn.functional.normalize(sigm(x), p=1)
        return ret

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_mod_stack = nn.Sequential(
            nn.Linear(6, 8),
            nn.Tanh(),
            nn.Linear(8,8),
            nn.Tanh(),
            nn.Linear(8,4),
            NormedSigmoid(),
        )
    def forward(self, x):
        return self.linear_tanh_mod_stack(x)

def train_loop(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_func(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_func):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_func(pred, y)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\nAccuracy: {(100*correct):>0.1f}%, Avg. loss: {test_loss:>8f}\n")

torch.set_default_dtype(torch.float64)
train_set = CustomTrainSet("./tt_data/train_annotations.csv", transform=normalize, target_transform=translate)
test_set = CustomTrainSet("./tt_data/test_annotations.csv", transform=normalize, target_transform=translate)

train_dataloader = DataLoader(train_set, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=100, shuffle=True)

model = NeuralNetwork()

batch_size = 100
epochs = 2

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.8)

for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------------")
    train_loop(train_dataloader, model, loss_func, optimizer)
    test_loop(test_dataloader, model, loss_func)
torch.save(model.state_dict(), "model_weights.pth")

for name, param in model.named_parameters():
    print(f"Name: {name}\nParams: {param.data.numpy()}")
    
params = iter(model.parameters())
W1 = next(params).data
b1 = next(params).data
W2 = next(params).data
b2 = next(params).data
W3 = next(params).data
b3 = next(params).data
np.savez("model_weights", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

print("\nDone. Saved weights to model_weights.pth and model_weights.npz")