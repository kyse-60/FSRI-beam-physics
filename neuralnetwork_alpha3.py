import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from time import perf_counter

dataset = pd.read_csv("output4eq-time-6.9866809000086505s.csv")
dataset = dataset.drop(columns=["time"]).sample(frac=1).reset_index(drop=True) #randomizes order, drops the time column
def natural_sort_key(x):
    prefix = x.rstrip('0123456789')
    number = x[len(prefix):]

    if number:
        return (prefix, int(number))
    else:
        return (prefix, -1)
    
dataset = dataset.reindex(
    sorted(dataset.columns, key=natural_sort_key),
    axis=1
)

phi_vals = dataset.loc[:,"phi0":"phi199"].to_numpy()
K_vals = dataset.loc[:, "k0":"k199"].to_numpy()
xi_vals = np.linspace(0,1,199)

coords = dataset.loc[:,"coord0":"coord199"]
x_coords = []
y_coords = []
alpha_vals = []

for row in range(0,dataset.shape[0]):
        alpha = (dataset.loc[row, "F"] * dataset.loc[row, "length"] * dataset.loc[row, "length"]) / (2 * dataset.loc[row, "E"] * (( dataset.loc[row, "base"] *  dataset.loc[row, "height"] **3)/ 12))
        alpha_vals.append(alpha)
        x_row = []
        y_row = []
        for item in range(0,200):
            point = coords.iloc[row,item]
            point = point = point.strip("()")
            x, y = point.split(",")
            x_row.append(float(x))
            y_row.append(float(y))
        x_coords.append(x_row)
        y_coords.append(y_row)

#the split for our 243 rows is 6/2/1, or 162/54/27
#alphas
training_alpha = torch.tensor(alpha_vals[0:162], dtype=torch.float32).reshape(-1, 1)
validation_alpha = torch.tensor(alpha_vals[162:216], dtype=torch.float32).reshape(-1, 1)
test_alpha = torch.tensor(alpha_vals[216:243], dtype=torch.float32).reshape(-1, 1)

#xis
training_xis = torch.tensor(xi_vals[0:162], dtype=torch.float32).reshape(-1, 1)
validation_xis = torch.tensor(xi_vals[162:216], dtype=torch.float32).reshape(-1, 1)
test_xis = torch.tensor(xi_vals[216:243], dtype=torch.float32).reshape(-1, 1)

#phis 
training_phi = torch.tensor(phi_vals[0:162], dtype=torch.float32)
validation_phi = torch.tensor(phi_vals[162:216], dtype=torch.float32)
test_phi = torch.tensor(phi_vals[216:243], dtype=torch.float32)

#K
training_k = torch.tensor(K_vals[0:162], dtype=torch.float32)
validation_k = torch.tensor(K_vals[162:216], dtype=torch.float32)
test_k = torch.tensor(K_vals[216:243], dtype=torch.float32)

#x
training_x = torch.tensor(x_coords[0:162], dtype=torch.float32)
validation_x = torch.tensor(x_coords[162:216], dtype=torch.float32)
test_x = torch.tensor(x_coords[216:243], dtype=torch.float32)

#y
training_y = torch.tensor(y_coords[0:162], dtype=torch.float32)
validation_y = torch.tensor(y_coords[162:216], dtype=torch.float32)
test_y = torch.tensor(y_coords[216:243], dtype=torch.float32)

print(training_y[0])
# class NeuralNetwork(nn.Module): #2 input, 11 outputs 
#     def __init__(self):
#         super().__init__()

#         self.trunk = nn.Sequential(
#             nn.Linear(2,32),
#             nn.ReLU(),
#             nn.Linear(32,32),
#             nn.ReLU(),
#         )
#         self.x_head = nn.Linear(32,1)
#         self.y_head = nn.Linear(32,1)
#         self.phi = nn.Linear(32,1)
#         self.K_val = nn.Linear(32,1) 

#     def forward(self, x):
#         features = self.trunk(x)
#         return self.x_head(features), self.y_head(features)

# model = NeuralNetwork()
# loss_function = nn.MSELoss()
# learning_rate = 0.001 #basically step size during each step in the training process
# optimizer = optim.Adam(model.parameters(), learning_rate)

# # TRAINING 
# num_epochs = 200 #100
# batch_size = 10  #10

# for epoch in range(num_epochs):
#     start = perf_counter()
#     for i in range(0, len(training_alpha), batch_size):
#         alpha_batch = training_alpha[i : i +batch_size]
#         xi_batch = training_xis[i : i +batch_size]

#         inputbatch = torch.cat([alpha_batch, xi_batch], dim =1)
#         x_pred, y_pred,phi_pred, k_pred = model(inputbatch)

#         xbatch = training_x[i: i +batch_size]
#         ybatch = training_y[i: i +batch_size]
#         phibatch = training_phi[i: i +batch_size]
#         kbatch = training_k[i: i +batch_size]

#         loss = loss_function(x_pred, xbatch) + loss_function(y_pred, ybatch) + loss_function(phi_pred, phibatch)+ loss_function(k_pred, kbatch)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     end = perf_counter()
#     epochtime = round(end-start, 3)
#     print(f'Epoch {epoch}: latest loss is {loss}, completed in {epochtime}s')

# #EVALUATING THE MODEL 
# with torch.no_grad():
#     validation_input = torch.cat([validation_alpha, validation_xis], dim =1)
#     x_pred, y_pred, phi_pred, k_pred = model(validation_input) #validation tensor so we can evluate instead of the training dataset

# accuracy = ((x_pred.round() == validation_x.round()).float().mean().item() 
#             + (y_pred.round() == validation_y.round()).float().mean().item()
#             +(phi_pred.round() == validation_phi.round()).float().mean().item()
#             + (k_pred.round() == validation_k.round()).float().mean().item())/4


# print(f'Accuracy {accuracy}')