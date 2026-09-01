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
xi_vals = np.linspace(0,1,200)

coords = dataset.loc[:,"coord0":"coord199"]
x_coords = []
y_coords = []
alpha_vals = []

for row in range(0,dataset.shape[0]):
        L = dataset.loc[row, "length"]
        alpha = (dataset.loc[row, "F"] * dataset.loc[row, "length"] * dataset.loc[row, "length"]) / (2 * dataset.loc[row, "E"] * (( dataset.loc[row, "base"] *  dataset.loc[row, "height"] **3)/ 12))
        alpha_vals.append(alpha)
        x_row = []
        y_row = []
        for item in range(0,200):
            point = coords.iloc[row,item]
            point = point = point.strip("()")
            x, y = point.split(",")
            x_row.append(float(x) / L)
            y_row.append(float(y) / L)
        x_coords.append(x_row)
        y_coords.append(y_row)
K_vals = K_vals * dataset["length"].to_numpy().reshape(-1, 1) 

#the split for our 243 rows is 6/2/1, or 162/54/27
#alphas                                                     
training_alpha = torch.tensor(np.repeat(alpha_vals[0:162], 200), dtype=torch.float32).reshape(-1, 1)
validation_alpha = torch.tensor(np.repeat(alpha_vals[162:216], 200), dtype=torch.float32).reshape(-1, 1)
test_alpha = torch.tensor(np.repeat(alpha_vals[216:243], 200), dtype=torch.float32).reshape(-1, 1)

#xi
training_xis = torch.tensor(np.tile(xi_vals, 162), dtype=torch.float32).reshape(-1, 1)
validation_xis = torch.tensor(np.tile(xi_vals, 54), dtype=torch.float32).reshape(-1, 1)
test_xis = torch.tensor(np.tile(xi_vals, 27), dtype=torch.float32).reshape(-1, 1)

training_phi = torch.tensor(phi_vals[0:162], dtype=torch.float32).flatten()
validation_phi = torch.tensor(phi_vals[162:216], dtype=torch.float32).flatten()
test_phi = torch.tensor(phi_vals[216:243], dtype=torch.float32).flatten()

#K
training_k = torch.tensor(K_vals[0:162], dtype=torch.float32).flatten()
validation_k = torch.tensor(K_vals[162:216], dtype=torch.float32).flatten()
test_k = torch.tensor(K_vals[216:243], dtype=torch.float32).flatten()

#x
training_x = torch.tensor(x_coords[0:162], dtype=torch.float32).flatten()
validation_x = torch.tensor(x_coords[162:216], dtype=torch.float32).flatten()
test_x = torch.tensor(x_coords[216:243], dtype=torch.float32).flatten()

#y
training_y = torch.tensor(y_coords[0:162], dtype=torch.float32).flatten()
validation_y = torch.tensor(y_coords[162:216], dtype=torch.float32).flatten()
test_y = torch.tensor(y_coords[216:243], dtype=torch.float32).flatten()

validation_output = [validation_x, validation_y, validation_phi, validation_k]
CASES= [1,2,3,4,6,8,10,20,30,40,50,60,70,80,90,100]

class NeuralNetwork(nn.Module):
    def __init__(self, n_hidden=128,n_layer = 5):
        super().__init__()

        layers = [nn.Linear(2,n_hidden), nn.Tanh()]
        for _ in range(n_layer - 2):
            layers += [nn.Linear(n_hidden,n_hidden),
                    nn.Tanh(),]
            layers += [nn.Linear(n_hidden,4)]
            self.net = nn.Sequential(*layers)

    def forward(self,xipts,alphapts):
        inputs = torch.stack([xipts,alphapts / lambda_scale], dim = 1)
        raw = self.input(inputs)
        xi = xipts[:, 0]
        x_pos = xi * raw[:,0]
        y_pos = xi * raw[:,1]
        phi = xi * raw[:,2]
        k = (1-xi) * raw[:,3]
        return torch.stack([x_pos,y_pos,phi,k], dim = 1)

model = NeuralNetwork()
loss_function = nn.MSELoss()
learning_rate = 0.001 #basically step size during each step in the training process
optimizer = optim.Adam(model.parameters(), learning_rate)
# 
# TRAINING 
num_epochs = 200 #100
batch_size = 10  #10     

def accuracy(predict, truth):
    accuracy = ((predict[0].round() == truth[0].round()).float().mean().item() 
            + (predict[1].round() == truth[1].round()).float().mean().item()
            +(predict[2].round() == truth[2].round()).float().mean().item()
            + (predict[3].round() == truth[3].round()).float().mean().item())/4
    return accuracy

df = pd.DataFrame()

for CASE in CASES:
    #reinit
    model = NeuralNetwork()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    #slicing it
    num = CASE * 200
    case_alpha = training_alpha[0:num]
    case_xis = training_xis[0:num]
    case_x = training_x[0:num]
    case_y = training_y[0:num]
    case_phi = training_phi[0:num]
    case_k = training_k[0:num]
    #train the model:
    start = perf_counter()
    for epoch in range(num_epochs):
        for i in range(0, len(case_alpha), batch_size):
            alpha_batch = case_alpha[i : i +batch_size]
            xi_batch = case_xis[i : i +batch_size]

            # inputbatch = torch.cat([alpha_batch, xi_batch], dim =1)
            pred = model(xi_batch,alpha_batch) 
            x_pred, y_pred, phi_pred, k_pred = pred.unbind(dim=1)

            xbatch = case_x[i: i +batch_size]
            ybatch = case_y[i: i +batch_size]
            phibatch = case_phi[i: i +batch_size]
            kbatch = case_k[i: i +batch_size]

            loss = loss_function(x_pred, xbatch) + loss_function(y_pred, ybatch) + loss_function(phi_pred, phibatch)+ loss_function(k_pred, kbatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #calculatingaccuracy
        # with torch.no_grad():
            # validation_input = torch.cat([validation_alpha, validation_xis], dim =1)
            # predicted = model(validation_input) 
            # val_mse = loss_function(predicted, torch.stack([validation_x, validation_y, validation_phi, validation_k], dim=1)).item() 
        # print(f'Epoch {epoch}: latest loss is {loss}, completed in {epochtime}s, val MSE {val_mse}')
    #evaluate its accuracy on a larger dataset
    end = perf_counter()
    traintime = round(end-start, 3)
    with torch.no_grad():
        # validation_input = torch.cat([validation_alpha, validation_xis], dim =1)
        predicted = model(validation_alpha, validation_xis) 
        val_mse = loss_function(predicted, torch.stack([validation_x, validation_y, validation_phi, validation_k], dim=1)).item() 
        acc = accuracy(predicted.unbind(dim=1), validation_output)
    print(f'Case number {CASE} is done!')
    df = pd.concat([df, pd.DataFrame([{'CASE': CASE,'MSE-loss': val_mse,'accuracy': acc,'traintime': traintime,}])], ignore_index=True)

df.to_csv("CaseIterationWBOUND.csv", index= False)