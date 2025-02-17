import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.signal import butter, lfilter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Allow duplicate OpenMP libraries without error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Use a single CPU thread for PyTorch
torch.set_num_threads(1)

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Function to apply a low-pass Butterworth filter
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff freq for filter design
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Define a multi-layer perceptron (MLP) model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super(MLPModel, self).__init__()
        # 1st fully-connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 2nd fully-connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 3rd fully-connected layer
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 4th fully-connected layer
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.bn4 = nn.BatchNorm1d(hidden_size * 4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # 5th fully-connected layer
        self.fc5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.bn5 = nn.BatchNorm1d(hidden_size * 2)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        
        # 6th fully-connected layer
        self.fc6 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc7 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through each layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.dropout4(out)
        
        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.dropout5(out)
        
        out = self.fc6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.dropout6(out)
        
        out = self.fc7(out)
        return out

# Early stopping mechanism to stop training when validation loss doesn't improve
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience      # How many epochs to wait before stopping
        self.min_delta = min_delta    # Minimum change in loss to be considered improvement
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # Check if this is the first time or if the new val_loss is significantly lower
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Name of the model file to save
model_name = "./MLP/model/mlp_model_generalized.pth"

# Directory containing your data files
file_dir = './'
file_names = [
    "motion_data_with_ground_truth_-0.2_15.csv",
    "motion_data_with_ground_truth_-0.2_30.csv",
    "motion_data_with_ground_truth_-0.2_45.csv",
    "motion_data_with_ground_truth_-0.25_15.csv",
    "motion_data_with_ground_truth_-0.25_30.csv",
    "motion_data_with_ground_truth_-0.25_45.csv",
    "motion_data_with_ground_truth_-0.3_15.csv",
    "motion_data_with_ground_truth_-0.3_30.csv",
    "motion_data_with_ground_truth_-0.3_45.csv",
    "motion_data_with_ground_truth_-0.35_15.csv",
    "motion_data_with_ground_truth_-0.35_30.csv",
    "motion_data_with_ground_truth_-0.35_45.csv"
]

# Lists to collect all features (X) and labels (y) from multiple files
all_X_tensor = []
all_y_tensor = []

# Loop through each file, load data, process, and accumulate
for file_name in file_names:
    file_path = os.path.join(file_dir, file_name)

    data = pd.read_csv(file_path)

    # Low-pass filter parameters
    cutoff_freq = 5.0
    fs = 100
    filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
    for col in filtered_columns:
        if col in data.columns:
            # Apply the low-pass filter to each relevant column
            data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

    # Choose filtered features and the label
    features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
    label = 'gait_phase'

    X = data[features].values
    y = data[label].values

    # Convert data to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_phase_original = torch.tensor(y, dtype=torch.float32)

    # Convert gait phase to sin/cos for better circular representation
    angle = 2 * math.pi * y_phase_original
    y_cos = torch.cos(angle).unsqueeze(-1)  # [N,1]
    y_sin = torch.sin(angle).unsqueeze(-1)  # [N,1]
    y_tensor = torch.cat([y_cos, y_sin], dim=-1)  # [N,2]

    # Normalize input features to [0,1]
    X_min = X_tensor.min(dim=0, keepdim=True).values
    X_max = X_tensor.max(dim=0, keepdim=True).values
    X_tensor = (X_tensor - X_min) / (X_max - X_min)

    # Use a sliding window approach for sequence data
    window_size = 10
    X_windowed = []
    y_windowed = []
    num_samples = X_tensor.shape[0]

    # Build sequences using the window size
    for i in range(num_samples - window_size + 1):
        # Flatten the window into one vector
        X_windowed.append(X_tensor[i:i+window_size].reshape(-1))
        # Label is the phase at the end of the window
        y_windowed.append(y_tensor[i+window_size-1])

    all_X_tensor.append(torch.stack(X_windowed))
    all_y_tensor.append(torch.stack(y_windowed))

# Concatenate all data from different files
X_tensor = torch.cat(all_X_tensor, dim=0)
y_tensor = torch.cat(all_y_tensor, dim=0)

print(f"Final X_tensor shape: {X_tensor.shape}")
print(f"Final y_tensor shape: {y_tensor.shape}")

# Create a dataset and split it into training and validation sets
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Model configuration
input_size = len(features) * 10  # 10 is the window_size
hidden_size = 128
output_size = 2

# Define DataLoaders for training and validation
batch_size = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define learning rate and create model
learning_rate = 0.001
model = MLPModel(input_size, hidden_size, output_size).to(device)

# Smooth L1 loss is less sensitive to outliers than MSE
criterion = nn.SmoothL1Loss()

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler to reduce LR when a metric has stopped improving
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    threshold=1e-3,
    threshold_mode='rel',
    cooldown=0,
    min_lr=1e-12
)

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    # Training phase
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.8f}, "
          f"Val Loss: {val_loss:.8f}, LR: {current_lr:.12f}")

    # Check for early stopping
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Save the trained model
model_save_path = model_name
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
