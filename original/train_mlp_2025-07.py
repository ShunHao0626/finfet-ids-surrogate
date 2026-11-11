import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Define the enhanced MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6, 100),  # Increased neurons for better capacity
            nn.BatchNorm1d(100),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Slightly increased dropout
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(50, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.features(x).squeeze(-1)

# Load JSON data
with open('N15A-nmos1.json', 'r') as f:
    data = json.load(f)

records = data['Records']
corner_dict = {}
corner_counter = 0
input_data = []
output_data = []

for record in records:
    Nfin = record['Nfin']
    L = record['L']
    Corner = record['Corner']
    Temp = record['Temp']
    Vds = record['Vds']
    Vgs_list = record['Vgs']
    Ids_list = record['Ids']
    if Corner not in corner_dict:
        corner_dict[Corner] = corner_counter
        corner_counter += 1
    corner_code = corner_dict[Corner]
    for Vgs, Ids in zip(Vgs_list, Ids_list):
        input_data.append([Nfin, L, corner_code, Temp, Vds, Vgs])
        output_data.append(Ids)

input_data = np.array(input_data)
output_data = np.array(output_data)

# Scale inputs and outputs
scaler_input = StandardScaler()
input_data[:, :5] = scaler_input.fit_transform(input_data[:, :5])

# Normalize output data
scaler_output = StandardScaler()
output_data = scaler_output.fit_transform(output_data.reshape(-1, 1)).flatten()

X = torch.tensor(input_data, dtype=torch.float32)
Y = torch.tensor(output_data, dtype=torch.float32)  # Use normalized output

train_size = int(0.9 * len(X))
train_dataset = TensorDataset(X[:train_size], Y[:train_size])
test_dataset = TensorDataset(X[train_size:], Y[train_size:])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MLP().to(device)
loss_function = nn.SmoothL1Loss()  # Huber Loss
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)  # Reduced weight decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # Cosine annealing schedule

# Training loop without early stopping
epochs = 2000
best_loss = float('inf')
patience = 50
counter = 0

train_losses, val_losses = [], []

for epoch in range(epochs):
    net.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = net(x_batch)
        loss = loss_function(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证集评估
    net.eval()
    val_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = net(x_batch)
            val_loss += loss_function(pred, y_batch).item() * len(x_batch)
            preds.append(pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    val_loss /= len(test_dataset)
    r2 = r2_score(np.concatenate(trues), np.concatenate(preds))

    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.6f} - Val MSE: {val_loss:.6f} - R2: {r2:.4f}")

    scheduler.step()

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(net.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


# Function to predict using the trained model and plot the fitting curve
def plot_fitting_curve(train_loader, model, scaler_input, scaler_output, save_path):
    model.eval()
    all_preds = []
    all_actual = []

    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch).cpu().numpy()  # Predict using the trained model
            actual = y_batch.cpu().numpy()  # Actual values from training set

            all_preds.extend(preds)
            all_actual.extend(actual)

    # Inverse transform to get the actual and predicted values in their original scale
    all_preds = scaler_output.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_actual = scaler_output.inverse_transform(np.array(all_actual).reshape(-1, 1)).flatten()

    # Plotting the comparison between actual and predicted values
    plt.figure(figsize=(8,6))

    # Scatter plot for actual vs predicted
    plt.scatter(all_actual, all_preds, color='blue', alpha=0.5, label="Actual vs Predicted")

    # Add a line y = x for better comparison (ideal fit line)
    plt.plot([min(all_actual), max(all_actual)], [min(all_actual), max(all_actual)], 'r--', label="Ideal Fit (y=x)")

    # Plot labels and title
    plt.xlabel("Actual Ids (A)")
    plt.ylabel("Predicted Ids (A)")
    plt.title("Model Fitting: Actual vs Predicted")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Plot ID-VD curve with smaller step size for smoother curve
def plot_idvd(fixed_params, vds_range, model, scaler_input, scaler_output, save_path):
    vds_list = np.linspace(*vds_range, num=200)
    inputs = []
    for vds in vds_list:
        vec = [fixed_params['Nfin'], fixed_params['L'], corner_dict[fixed_params['Corner']],
               fixed_params['Temp'], vds, fixed_params['Vgs']]
        inputs.append(vec)
    inputs = np.array(inputs)
    inputs[:, :5] = scaler_input.transform(inputs[:, :5])
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(inputs).cpu().numpy()

    ids = scaler_output.inverse_transform(preds.reshape(-1, 1)).flatten()
    # 🔴 clip removed here
    # ids = np.clip(ids, 0, 2e-3)

    plt.figure()
    plt.plot(vds_list, ids, label=f"Vgs={fixed_params['Vgs']}V")
    plt.xlabel("Vds (V)")
    plt.ylabel("Ids (A)")
    plt.title("ID-VD Curve (Raw Output)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# Plot ID-VG curve with smaller step size for smoother curve
def plot_idvg(fixed_params, vgs_range, model, scaler_input, scaler_output, save_path):
    vgs_list = np.linspace(*vgs_range, num=200)
    inputs = []
    for vgs in vgs_list:
        vec = [fixed_params['Nfin'], fixed_params['L'], corner_dict[fixed_params['Corner']],
               fixed_params['Temp'], fixed_params['Vds'], vgs]
        inputs.append(vec)
    inputs = np.array(inputs)
    inputs[:, :5] = scaler_input.transform(inputs[:, :5])
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(inputs).cpu().numpy()

    ids = scaler_output.inverse_transform(preds.reshape(-1, 1)).flatten()
    # 🔴 clip removed here
    # ids = np.clip(ids, 0, 2e-3)

    plt.figure()
    plt.plot(vgs_list, ids, label=f"Vds={fixed_params['Vds']}V")
    plt.xlabel("Vgs (V)")
    plt.ylabel("Ids (A)")
    plt.title("ID-VG Curve (Raw Output)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# After training the model, call the function to plot the three curves
fixed_params = {'Nfin': 1, 'L': 15, 'Corner': 'tt', 'Temp': 25, 'Vgs': 0.7}
plot_idvd(fixed_params, vds_range=(0, 1.0), model=net, scaler_input=scaler_input, scaler_output=scaler_output, save_path="idvd_curve.png")

fixed_params = {'Nfin': 1, 'L': 15, 'Corner': 'tt', 'Temp': 25, 'Vds': 0.9}
plot_idvg(fixed_params, vgs_range=(0, 1.0), model=net, scaler_input=scaler_input, scaler_output=scaler_output, save_path="idvg_curve.png")

# Plot the fitting curve
plot_fitting_curve(train_loader, net, scaler_input, scaler_output, save_path="model_fitting_curve.png")