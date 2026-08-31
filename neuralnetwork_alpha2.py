import pandas as pd
import torch
from torch import nn
from torch import optim
from time import perf_counter

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
        for item in range(0,11):
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
training_x = torch.tensor(x_coords[0:600], dtype=torch.float32)
validation_x = torch.tensor(x_coords[600:900], dtype=torch.float32)
test_x = torch.tensor(x_coords[900:1001], dtype=torch.float32)
training_y = torch.tensor(y_coords[0:600], dtype=torch.float32)
validation_y = torch.tensor(y_coords[600:900], dtype=torch.float32)
test_y = torch.tensor(y_coords[900:1001], dtype=torch.float32)

# Epoch 0: latest loss is 0.0008906425791792572, completed in 0.08s
# Epoch 1: latest loss is 0.00012630947458092123, completed in 0.077s
# Epoch 2: latest loss is 5.490777766681276e-05, completed in 0.077s
# Epoch 3: latest loss is 3.8113528717076406e-05, completed in 0.079s
# Epoch 4: latest loss is 2.2147150957607664e-05, completed in 0.085s
# Epoch 5: latest loss is 1.5075408555276226e-05, completed in 0.075s
# Epoch 6: latest loss is 1.3332289199752267e-05, completed in 0.08s
# Epoch 7: latest loss is 1.3463485629472416e-05, completed in 0.114s
# Epoch 8: latest loss is 1.1930874279642012e-05, completed in 0.078s
# Epoch 9: latest loss is 8.554461601306684e-06, completed in 0.076s
# Accuracy 0.09090909361839294

class NeuralNetwork(nn.Module): #1 input, 11 outputs 
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
        )
        self.x_head = nn.Linear(32,11)
        self.y_head = nn.Linear(32,11)

    def forward(self, x):
        features = self.trunk(x)
        return self.x_head(features), self.y_head(features)

model = NeuralNetwork()
loss_function = nn.MSELoss()
learning_rate = 0.001 #basically step size during each step in the training process
optimizer = optim.Adam(model.parameters(), learning_rate)

# TRAINING 
num_epochs = 200 #100
batch_size = 10  #10

for epoch in range(num_epochs):
    start = perf_counter()
    for i in range(0, len(training_input), batch_size):
        inputbatch = training_input[i : i +batch_size]
        x_pred, y_pred = model(inputbatch)

        xbatch = training_y[i: i +batch_size]
        ybatch = training_y[i: i +batch_size]

        loss = loss_function(x_pred, xbatch) + loss_function(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = perf_counter()
    epochtime = round(end-start, 3)
    print(f'Epoch {epoch}: latest loss is {loss}, completed in {epochtime}s')

#EVALUATING THE MODEL 
with torch.no_grad():
    x_pred, y_pred = model(validation_input) #validation tensor so we can evluate instead of the training dataset

accuracy = ((x_pred.round() == validation_x.round()).float().mean().item() + (y_pred.round() == validation_y.round()).float().mean().item())/2


print(f'Accuracy {accuracy}')