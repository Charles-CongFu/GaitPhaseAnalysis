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


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

# Low-pass filter function using Butterworth filter
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Positional encoding module for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Prepare a positional encoding table
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position.float() * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        seq_length = x.size(1)
        # Add positional encoding up to the current sequence length
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)

# Transformer encoder model
class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.0,
        output_size=2
    ):
        super(TransformerEncoderModel, self).__init__()
        # Project input_dim to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Define a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer to produce output_size dimensions
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        out = self.fc_out(encoded)
        return out

# Early stopping helper class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience      # Number of epochs to wait
        self.min_delta = min_delta    # Minimum improvement threshold
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # If this is the first loss or if improvement is higher than threshold
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

early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

# Model file name to save
model_name = "./Transformer_encoder/model/transformer_encoder_model_generailized.pth"

# Directory containing data files
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

# Lists to store sequences of inputs (X) and labels (y)
all_X_tensor_seq = []
all_y_tensor_seq = []

# Load and process each file
for file_name in file_names:
    file_path = os.path.join(file_dir, file_name)
    
    data = pd.read_csv(file_path)

    # Filter columns
    cutoff_freq = 5.0
    fs = 100
    filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
    for col in filtered_columns:
        if col in data.columns:
            # Apply low-pass filter
            data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

    # Select features and label
    features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
    label = 'gait_phase'

    X = data[features].values
    y = data[label].values

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_phase_original = torch.tensor(y, dtype=torch.float32)

    # Convert the gait phase to sin/cos format
    angle = 2 * math.pi * y_phase_original
    y_cos = torch.cos(angle).unsqueeze(-1)  # shape: [N,1]
    y_sin = torch.sin(angle).unsqueeze(-1)  # shape: [N,1]
    y_tensor = torch.cat([y_cos, y_sin], dim=-1)  # shape: [N,2]

    # Normalize features to [0,1]
    X_min = X_tensor.min(dim=0, keepdim=True).values
    X_max = X_tensor.max(dim=0, keepdim=True).values
    X_tensor = (X_tensor - X_min) / (X_max - X_min)

    # Define sequence length
    seq_length = 10
    num_samples = X_tensor.shape[0]
    input_dim = X_tensor.shape[1]

    X_seq_list = []
    y_seq_list = []

    # Build sequences of length seq_length
    for i in range(num_samples - seq_length + 1):
        # X_seq shape: [seq_length, input_dim]
        X_seq = X_tensor[i:i+seq_length]
        # y_seq shape: [seq_length, 2]
        y_seq = y_tensor[i:i+seq_length]
        
        # Add a batch dimension: shape [1, seq_length, input_dim]
        X_seq_list.append(X_seq.unsqueeze(0))
        # shape [1, seq_length, 2]
        y_seq_list.append(y_seq.unsqueeze(0))

    # Combine all sequences into tensors
    if X_seq_list and y_seq_list:
        all_X_tensor_seq.append(torch.cat(X_seq_list, dim=0))
        all_y_tensor_seq.append(torch.cat(y_seq_list, dim=0))

# Concatenate all sequence data from all files
X_tensor_seq = torch.cat(all_X_tensor_seq, dim=0)  # [total_seq_count, seq_length, input_dim]
y_tensor_seq = torch.cat(all_y_tensor_seq, dim=0)  # [total_seq_count, seq_length, 2]

print(f"Final X_tensor_seq shape: {X_tensor_seq.shape}")
print(f"Final y_tensor_seq shape: {y_tensor_seq.shape}")

# Create a dataset and split into training and validation
dataset = TensorDataset(X_tensor_seq, y_tensor_seq)
num_sequences = X_tensor_seq.shape[0]

train_size = int(0.8 * num_sequences)
val_size = num_sequences - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the Transformer model
model = TransformerEncoderModel(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.0,
    output_size=2
).to(device)

criterion = nn.SmoothL1Loss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Reduce learning rate if validation loss plateaus
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

num_epochs = 2000

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

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

    scheduler.step(val_loss)
    # early_stopping(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.12f}, "
          f"Val Loss: {val_loss:.12f}, LR: {current_lr:.12f}")

    # Check if early stopping is triggered
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Save the trained model
model_save_path = model_name
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
