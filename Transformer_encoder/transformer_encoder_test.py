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
import time

# Allow duplicates of OpenMP libraries without error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Restrict PyTorch to a single CPU thread
torch.set_num_threads(1)

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define a low-pass filter function using Butterworth filter
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs              # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Define positional encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        # Fill even indices with sin and odd indices with cos
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        pe = pe.unsqueeze(0)         # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        seq_length = x.size(1)
        # Add positional encoding to input
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)

# Define a Transformer Encoder Model
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.0, output_size=2):
        super(TransformerEncoderModel, self).__init__()
        # Project input to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Define a Transformer encoder layer and stack multiple layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer to produce the desired output size
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        out = self.fc_out(encoded)
        return out

# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # If this is the first recorded loss or if the new loss is better than the threshold
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

# Models and data can be replaced as needed
model_name = "./Transformer_encoder/model/transformer_encoder_model_generailized.pth"
file_dir = './'
file_names = [
    # "motion_data_with_ground_truth_-0.2_15.csv",
    # "motion_data_with_ground_truth_-0.2_30.csv",
    # "motion_data_with_ground_truth_-0.2_45.csv",
    # "motion_data_with_ground_truth_-0.25_15.csv",
    # "motion_data_with_ground_truth_-0.25_30.csv",
    # "motion_data_with_ground_truth_-0.25_45.csv",
    # "motion_data_with_ground_truth_-0.3_15.csv",
    # "motion_data_with_ground_truth_-0.3_30.csv",
    # "motion_data_with_ground_truth_-0.3_45.csv",
    # "motion_data_with_ground_truth_-0.35_15.csv",
    # "motion_data_with_ground_truth_-0.35_30.csv",
    # "motion_data_with_ground_truth_-0.35_45.csv",
    "motion_data_with_ground_truth_test_0.275_30.csv",
    # Additional files can be added here as needed
]

# Lists to collect sequences for input (X) and labels (y)
all_X_tensor_seq = []
all_y_tensor_seq = []

# Load and process each file
for file_name in file_names:
    file_path = os.path.join(file_dir, file_name)
    data = pd.read_csv(file_path)

    # Apply a low-pass filter to relevant columns
    cutoff_freq = 5.0
    fs = 100
    filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
    for col in filtered_columns:
        if col in data.columns:
            data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

    # Select features and label
    features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
    label = 'gait_phase'

    # Convert data to torch tensors
    X = data[features].values
    y = data[label].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_phase_original = torch.tensor(y, dtype=torch.float32)

    # Convert the gait phase to sin/cos format
    angle = 2 * math.pi * y_phase_original
    y_cos = torch.cos(angle).unsqueeze(-1)  # shape: [N, 1]
    y_sin = torch.sin(angle).unsqueeze(-1)  # shape: [N, 1]
    y_tensor = torch.cat([y_cos, y_sin], dim=-1)  # shape: [N, 2]

    # Normalize features to [0,1]
    X_min = X_tensor.min(dim=0, keepdim=True).values
    X_max = X_tensor.max(dim=0, keepdim=True).values
    X_tensor = (X_tensor - X_min) / (X_max - X_min)

    # Set sequence length
    seq_length = 10
    num_samples = X_tensor.shape[0]
    input_dim = X_tensor.shape[1]

    X_seq_list = []
    y_seq_list = []

    # Create sequences of length seq_length
    for i in range(num_samples - seq_length + 1):
        # Slices for a single sequence
        X_seq = X_tensor[i:i+seq_length]       # shape: [seq_length, input_dim]
        y_seq = y_tensor[i:i+seq_length]       # shape: [seq_length, 2]
        
        # Add a batch dimension
        X_seq_list.append(X_seq.unsqueeze(0))
        y_seq_list.append(y_seq.unsqueeze(0))

    # Concatenate all sequences if any were created
    if X_seq_list and y_seq_list:
        all_X_tensor_seq.append(torch.cat(X_seq_list, dim=0))
        all_y_tensor_seq.append(torch.cat(y_seq_list, dim=0))

# Combine all sequence data from all files
X_tensor_seq = torch.cat(all_X_tensor_seq, dim=0)  # shape: [total_num_sequences, seq_length, input_dim]
y_tensor_seq = torch.cat(all_y_tensor_seq, dim=0)  # shape: [total_num_sequences, seq_length, 2]

print(f"Final X_tensor_seq shape: {X_tensor_seq.shape}")
print(f"Final y_tensor_seq shape: {y_tensor_seq.shape}")

# Initialize the Transformer encoder model
model = TransformerEncoderModel(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.0,
    output_size=2
).to(device)

# Paths to load a pre-trained model
model_save_path = model_name
model_load_path = model_save_path

# Create a second instance of TransformerEncoderModel with the same config
loaded_model = TransformerEncoderModel(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.0,
    output_size=2
).to(device)

# Load saved model weights
loaded_model.load_state_dict(torch.load(model_load_path, map_location=device))
loaded_model.eval()
print(f"Model loaded from {model_load_path}")

# Prepare data for inference
# X_tensor_seq shape: [num_sequences, seq_length, input_dim]
# Take only the last element of each sequence for y, shape becomes [num_sequences, 2]
y_tensor_seq = y_tensor_seq[:, -1, :]  # shape: [num_sequences, 2]

# Create a DataLoader for validation/inference
dataset = TensorDataset(X_tensor_seq, y_tensor_seq)
val_loader_seq = DataLoader(dataset, batch_size=1, shuffle=False)

y_pred_list = []
y_actual_list = []
total_time = 0.0

# Perform inference
with torch.no_grad():
    start_time = time.time()
    for batch_X, batch_y in val_loader_seq:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        pred = loaded_model(batch_X)
        # Collect the last element of each sequence in the batch
        y_pred_list.append(pred[:, -1, :].cpu().numpy())
        y_actual_list.append(batch_y.cpu().numpy())
    end_time = time.time()
    total_time += (end_time - start_time)

# Print the average inference time based on 9991 samples
print(total_time / 9991)

# Convert list of predictions/labels to NumPy arrays
y_pred_seq = np.concatenate(y_pred_list, axis=0)   # shape: [num_sequences, 2]
y_actual_seq = np.concatenate(y_actual_list, axis=0)  # shape: [num_sequences, 2]

# Flatten original gait phase for plotting reference
y_phase_val = y_phase_original.reshape(-1).numpy()

# Convert predicted sin/cos to angle in [0,1]
pred_angle = np.arctan2(y_pred_seq[:, 1], y_pred_seq[:, 0])
pred_phase = (pred_angle / (2 * math.pi)) % 1.0

# Convert actual sin/cos to angle in [0,1]
actual_angle = np.arctan2(y_actual_seq[:, 1], y_actual_seq[:, 0])
actual_phase = (actual_angle / (2 * math.pi)) % 1.0

y_pred = pred_phase
y_actual = actual_phase

# Direct sin/cos differences
differences = np.abs(y_pred_seq - y_actual_seq)
max_difference_sin = np.max(differences[:, 1])
max_difference_cos = np.max(differences[:, 0])
min_difference_sin = np.min(differences[:, 1])
min_difference_cos = np.min(differences[:, 0])
mean_difference_sin = np.mean(differences[:, 1])
mean_difference_cos = np.mean(differences[:, 0])
sum = mean_difference_cos + mean_difference_sin
print("------- Circular Distance Results -------")
print(f"Maximum Difference (sin)/(cos): {max_difference_sin} /  {max_difference_cos} ")
print(f"Minimum Difference (sin)/(cos): {min_difference_sin} /  {min_difference_cos}")
print(f"Mean Difference (sin)/(cos): {mean_difference_sin} /  {mean_difference_cos}")
print(f"Mean Difference (sin + cos): {sum}")

# Absolute phase differences (circular scale)
differences = np.abs(y_actual - y_pred)
max_difference_circular = np.max(differences)
min_difference_circular = np.min(differences)
mean_difference_circular = np.mean(differences)

print("------- Linear Distance with Offset Results -------")
print(f"Maximum Difference (circular): {max_difference_circular}")
print(f"Minimum Difference (circular): {min_difference_circular}")
print(f"Mean Difference (circular): {mean_difference_circular}")

# Define a function to handle phase wrap-around
def calculate_gait_phase_error_sample(ground_truth, prediction):
    if np.abs(ground_truth - prediction) < 0.5:
        return abs(ground_truth - prediction)
    elif ground_truth > prediction:
        return abs(ground_truth - (1 + prediction))
    else:
        return abs((1 + ground_truth) - prediction)

# Vectorize this function to handle arrays
calculate_gait_phase_error = np.vectorize(calculate_gait_phase_error_sample)
filtered_diffs = calculate_gait_phase_error(y_actual, y_pred)

max_diff = np.max(filtered_diffs)
min_diff = np.min(filtered_diffs)
mean_diff = np.mean(filtered_diffs)
print("------- New Linear Distance with Offset Results -------")
print(f"Maximum Difference (linear after offset): {max_diff}")
print(f"Minimum Difference (linear after offset): {min_diff}")
print(f"Mean Difference (linear after offset): {mean_diff}")

# Plotting: actual vs. predicted gait phase
visible_time_range = 500
fig_width_cm = visible_time_range / 50
fig_height = 8
fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height / 2.54

fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
current_start_time = 0

def update_plot(start_time, seq_length):
    ax.clear()
    end_time = start_time + visible_time_range

    # Actual data in the visible range
    actual_data = y_phase_val[int(start_time):int(end_time)]
    
    # Insert None for the first (seq_length-1) frames to align predictions
    predicted_data_shifted = [None] * (seq_length - 1) + list(y_pred)
    predicted_data = predicted_data_shifted[int(start_time):int(end_time)]

    ax.plot(actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)
    ax.plot(predicted_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gait Cycle Value', fontsize=12)
    ax.set_title('Transformer encoder', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.draw()

# Initial plot
update_plot(current_start_time, seq_length)

# Create a slider to scroll through the timeline
slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(
    slider_ax_time,
    'Time Scroll',
    0,
    max(0, len(y_actual) - visible_time_range),
    valinit=0,
    valstep=1
)

def on_time_slider_update(val):
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_time, seq_length)

slider_time.on_changed(on_time_slider_update)
plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])
plt.show()
