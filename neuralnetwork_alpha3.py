import pandas as pd
import torch
from torch import nn
from torch import optim
from time import perf_counter

dataset = pd.read_csv("outputValpha(1000).csv")
dataset = dataset.drop(columns=["time"]).sample(frac=1).reset_index(drop=True) #randomizes order, drops the time column

alpha_vals = dataset["alpha"]
xi_vals = dataset["xi"]
phi_vals =
K_vals = 

coords = dataset.loc[:,"coord0":"coord10", ]
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
# train for 1,2,3,4,5,..10,20,40,60,80,100 and record it (each one has 200 inside it because its 200 points correspponding to the alphas )
# should record the MSE and accuracy for each test 
# plot the graph MSE v samples
#this is a hard constraint (put in poster) or boundary embedded 

#for the bounds --> we know x(0) = y(0) = phi(0) = K(1) = 0
# whatever the network gives me multiply by xi 
'''
(x)(xi) = x
(y)(xi) = y
...


'''


#alphas
training_alpha = torch.tensor(alpha_vals.loc[0:599].to_numpy(), dtype=torch.float32).reshape(-1, 1)
validation_alpha = torch.tensor(alpha_vals.loc[600:899].to_numpy(), dtype=torch.float32).reshape(-1, 1)
test_alpha = torch.tensor(alpha_vals.loc[900:1000].to_numpy(), dtype=torch.float32).reshape(-1, 1)

#xis
training_xis = torch.tensor(xi_vals.loc[0:599].to_numpy(), dtype=torch.float32).reshape(-1, 1)
validation_xis = torch.tensor(xi_vals.loc[600:899].to_numpy(), dtype=torch.float32).reshape(-1, 1)
test_xis = torch.tensor(xi_vals.loc[900:1000].to_numpy(), dtype=torch.float32).reshape(-1, 1)

#phis 
training_phi = torch.tensor(phi_vals[0:600], dtype=torch.float32)
validation_phi = torch.tensor(phi_vals[600:900], dtype=torch.float32)
test_phi = torch.tensor(phi_vals[900:1001], dtype=torch.float32)

#K
training_k = torch.tensor(K_vals[0:600], dtype=torch.float32)
validation_k = torch.tensor(K_vals[600:900], dtype=torch.float32)
test_k = torch.tensor(K_vals[900:1001], dtype=torch.float32)

#x
training_x = torch.tensor(x_coords[0:600], dtype=torch.float32)
validation_x = torch.tensor(x_coords[600:900], dtype=torch.float32)
test_x = torch.tensor(x_coords[900:1001], dtype=torch.float32)

#y
training_y = torch.tensor(y_coords[0:600], dtype=torch.float32)
validation_y = torch.tensor(y_coords[600:900], dtype=torch.float32)
test_y = torch.tensor(y_coords[900:1001], dtype=torch.float32)

validation_output = [validation_x, validation_y, validation_phi, validation_k]

class NeuralNetwork(nn.Module): #2 input, 11 outputs 
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,4)
        )
        # self.x_head = nn.Linear(32,1)
        # self.y_head = nn.Linear(32,1)
        # self.phi = nn.Linear(32,1)
        # self.K_val = nn.Linear(32,1) 

    def forward(self, x):
        return self.trunk(x) #x,y,phi,k

model = NeuralNetwork()
loss_function = nn.MSELoss()
learning_rate = 0.001 #basically step size during each step in the training process
optimizer = optim.Adam(model.parameters(), learning_rate)

# TRAINING 
num_epochs = 200 #100
batch_size = 10  #10

def accuracy(predict, truth):
    accuracy = ((predict[0].round() == truth[0].round()).float().mean().item() 
            + (predict[1].round() == truth[1].round()).float().mean().item()
            +(predict[2].round() == truth[2].round()).float().mean().item()
            + (predict[3].round() == truth[3].round()).float().mean().item())/4
    return accuracy

for epoch in range(num_epochs):
    start = perf_counter()
    for i in range(0, len(training_alpha), batch_size):
        alpha_batch = training_alpha[i : i +batch_size]
        xi_batch = training_xis[i : i +batch_size]

        inputbatch = torch.cat([alpha_batch, xi_batch], dim =1)
        x_pred, y_pred,phi_pred, k_pred = model(inputbatch)

        xbatch = training_x[i: i +batch_size]
        ybatch = training_y[i: i +batch_size]
        phibatch = training_phi[i: i +batch_size]
        kbatch = training_k[i: i +batch_size]

        loss = loss_function(x_pred, xbatch) + loss_function(y_pred, ybatch) + loss_function(phi_pred, phibatch)+ loss_function(k_pred, kbatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = perf_counter()
    epochtime = round(end-start, 3)
    #calculatingaccuracy
    with torch.no_grad():
        validation_input = torch.cat([validation_alpha, validation_xis], dim =1)
        predicted = model(validation_input) 
        acc = accuracy(predicted,validation_output)
    print(f'Epoch {epoch}: latest loss is {loss}, completed in {epochtime}s, accuracy {acc}')

#EVALUATING THE MODEL

with torch.no_grad():
    validation_input = torch.cat([validation_alpha, validation_xis], dim =1)
    predicted = model(validation_input) 
    acc = accuracy(predicted,validation_output)

print(f'Accuracy {accuracy}')