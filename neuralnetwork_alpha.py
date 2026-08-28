import pandas as pd
import torch
from torch import nn
from torch import optim

dataset = pd.read_csv("outputValpha(1000).csv")
dataset = dataset.drop(columns=["time"]).sample(frac=1).reset_index(drop=True) #randomizes order, drops the time column

alpha_vals = dataset["alpha"]
#input = torch.tensor(alpha_vals, dtype=torch.float32).reshape(-1, 1)

coords = dataset.loc[:,"coord0":"coord10"]
x_coords = []
y_coords = []

for row in range(0,coords.shape[0]):
        x_row = []
        y_row = []
        for item in range(0,10):
            point = coords.iloc[row,item]
            point = point = point.strip("()")
            x, y = point.split(",")
            x_row.append(float(x))
            y_row.append(float(y))
        x_coords.append(x_row)
        y_coords.append(y_row)

# x_output = torch.tensor(x_coords, dtype=torch.float32)
# y_output = torch.tensor(y_coords, dtype=torch.float32)

training_input = torch.tensor(alpha_vals.loc[0:599].to_numpy(), dtype=torch.float32).reshape(-1, 1)
validation_input = torch.tensor(alpha_vals.loc[600:899].to_numpy(), dtype=torch.float32).reshape(-1, 1)
test_input = torch.tensor(alpha_vals.loc[900:1000].to_numpy(), dtype=torch.float32).reshape(-1, 1)
training_x = torch.tensor(x_coords[0:599], dtype=torch.float32)
validation_x = torch.tensor(x_coords[600:899], dtype=torch.float32)
test_x = torch.tensor(x_coords[900:1000], dtype=torch.float32)
training_y = torch.tensor(y_coords[0:599], dtype=torch.float32)
validation_y = torch.tensor(y_coords[600:899], dtype=torch.float32)
test_y = torch.tensor(y_coords[900:1000], dtype=torch.float32)

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
