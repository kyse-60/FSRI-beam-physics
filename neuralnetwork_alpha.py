import pandas as pd
import torch
from torch import nn
from torch import optim
from time import perf_counter

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

# TRAINING 
num_epochs = 10 #100
batch_size = 5  #10

for epoch in range(num_epochs):
    start = perf_counter()
    for i in range(0, len(input), batch_size):
        Xbatch = input[i : i +batch_size]
        y_pred = model(Xbatch)
        ybatch = output[i: i +batch_size]
        loss = loss_function(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = perf_counter()
    epochtime = round(end-start, 0.001)
    print(f'Epoch {epoch}: latest loss is {loss}, completed in {epochtime}s')

#EVALUATING THE MODEL 
with torch.no_grad():
    validation_pred = model(validation_input) #validation tensor so we can evluate instead of the training dataset

accuracy = (validation_pred.round() == validation_output).float().mean()

print(f'Accuracy {accuracy}')