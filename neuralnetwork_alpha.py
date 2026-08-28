import pandas as pd
import torch
from torch import nn
from torch import optim

dataset = pd.read_csv("outputValpha(1000).csv")
dataset = dataset.drop(columns=["time"]).sample(frac=1).reset_index(drop=True) #randomizes order, drops the time column
training_set = dataset.loc[0:599] #600 items
validation_set = dataset.loc[600:899] #300 items
test_set = dataset.loc[900:1000] #101 items

class NeuralNetwork(nn.Module): #1 input, 11 outputs 
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,11)
        )

    def forward(self, x):
        return self.network(x)

model = NeuralNetwork()

loss_function = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
