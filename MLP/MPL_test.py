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

# Allow duplicates of OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set PyTorch to use a single CPU thread
torch.set_num_threads(1)

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Low-pass filter function using Butterworth filter
def lowpass_filter(data, cutoff, fs, order=4):
    # nyquist frequency is half of the sampling rate
    nyquist = 0.5 * fs
    # normalized cutoff frequency for filter design
    normal_cutoff = cutoff / nyquist
    # design butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # apply the filter using lfilter
    return lfilter(b, a, data)

# Define an MLP model with multiple layers and optional dropout
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super(MLPModel, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third fully connected layer
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth fully connected layer
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.bn4 = nn.BatchNorm1d(hidden_size * 4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Fifth fully connected layer
        self.fc5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.bn5 = nn.BatchNorm1d(hidden_size * 2)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        
        # Sixth fully connected layer
        self.fc6 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout_rate)
        
        # Final output layer
        self.fc7 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass input through first layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Pass through second layer
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Pass through third layer
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        # Pass through fourth layer
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.dropout4(out)
        
        # Pass through fifth layer
        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.dropout5(out)
        
        # Pass through sixth layer
        out = self.fc6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.dropout6(out)
        
        # Final output layer
        out = self.fc7(out)
        return out

# Models and data can be replaced as needed
model_name = "./MLP/model/mlp_model_generalized.pth" 
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
    #find break
    # "motion_data_with_ground_truth_femur_l.csv",
    # "motion_data_with_ground_truth_femur_r.csv",
    # "motion_data_with_ground_truth_tibia_r.csv",
    # "motion_data_with_ground_truth_torso.csv"
]

# Lists to store processed inputs/outputs
all_X_tensor = []
all_y_tensor = []

for file_name in file_names:
    file_path = os.path.join(file_dir, file_name)
    
    # Read CSV data
    data = pd.read_csv(file_path)

    # Define filter params
    cutoff_freq = 5.0
    fs = 100
    filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
    for col in filtered_columns:
        if col in data.columns:
            # Apply low-pass filter
            data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

    # Define features and label
    features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
    label = 'gait_phase'

    # Extract feature (X) and label (y) arrays
    X = data[features].values
    y = data[label].values

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_phase_original = torch.tensor(y, dtype=torch.float32)

    # Convert gait phase to sin and cos representation
    angle = 2 * math.pi * y_phase_original
    y_cos = torch.cos(angle).unsqueeze(-1)  # [N,1]
    y_sin = torch.sin(angle).unsqueeze(-1)  # [N,1]
    y_tensor = torch.cat([y_cos, y_sin], dim=-1)  # [N,2]

    # Normalize input features to [0,1]
    X_min = X_tensor.min(dim=0, keepdim=True).values
    X_max = X_tensor.max(dim=0, keepdim=True).values
    X_tensor = (X_tensor - X_min) / (X_max - X_min)

    # Define window size for sliding window approach
    window_size = 10
    X_windowed = []
    y_windowed = []
    num_samples = X_tensor.shape[0]

    # Collect windowed samples
    for i in range(num_samples - window_size + 1):
        # Flatten window of features into a single vector
        X_windowed.append(X_tensor[i:i+window_size].reshape(-1))  # [window_size*feature_dim]
        # Label is the value at the end of the window
        y_windowed.append(y_tensor[i+window_size-1])

    # Convert lists to tensors and accumulate
    all_X_tensor.append(torch.stack(X_windowed))
    all_y_tensor.append(torch.stack(y_windowed))

# Concatenate all windowed data from different files
X_tensor = torch.cat(all_X_tensor, dim=0)
y_tensor = torch.cat(all_y_tensor, dim=0)

print(f"Final X_tensor shape: {X_tensor.shape}")
print(f"Final y_tensor shape: {y_tensor.shape}")

# Model parameters
input_size = len(features) * window_size
hidden_size = 128
output_size = 2

# Load the saved model
model_save_path = model_name
model_load_path = model_save_path
loaded_model = MLPModel(input_size, hidden_size, output_size).to(device)
loaded_model.load_state_dict(torch.load(model_load_path, map_location=device))
loaded_model.eval()
print(f"Model loaded from {model_load_path}")

# Measure inference time
total_time = 0.0
with torch.no_grad():
    start_time = time.time()
    X_tensor = X_tensor.to(device)
    # Predict sin/cos values
    y_pred = loaded_model(X_tensor).cpu().numpy()  # [M,2]
    y_actual = y_tensor.cpu().numpy()              # [M,2]
    end_time = time.time()
    total_time += (end_time - start_time)

# Print average time per inference
print(total_time/9991)

# Convert predicted sin/cos values to angles in [0,1]
pred_angle = np.arctan2(y_pred[:,1], y_pred[:,0])
pred_phase = (pred_angle / (2*math.pi)) % 1.0

# Ground truth (shifted by window size)
y_phase = y_phase_original[window_size-1:].numpy()

# Calculate direct differences of sin/cos
# This helps understand how well the model fits the circular representation

# Compute absolute differences
differences = np.abs(y_pred - y_actual)
max_difference_sin = np.max(differences[:,1])
max_difference_cos = np.max(differences[:,0])
min_difference_sin = np.min(differences[:,1])
min_difference_cos = np.min(differences[:,0])
mean_difference_sin = np.mean(differences[:,1])
mean_difference_cos = np.mean(differences[:,0])
sum = mean_difference_cos + mean_difference_sin
print("------- Circular Distance Results -------")
print(f"Maximum Difference (sin)/(cos): {max_difference_sin} /  {max_difference_cos} ")
print(f"Minimum Difference (sin)/(cos): {min_difference_sin} /  {min_difference_cos}")
print(f"Mean Difference (sin)/(cos): {mean_difference_sin} /  {mean_difference_cos}")
print(f"Mean Difference (sin + cos): {sum}")

# Calculate linear difference after converting to phase [0,1]
final_diffs = np.abs(y_phase - pred_phase)
max_difference_linear = np.max(final_diffs)
min_difference_linear = np.min(final_diffs)
mean_difference_linear = np.mean(final_diffs)
print("------- Linear Distance with Offset Results -------")
print(f"Maximum Difference (linear after offset): {max_difference_linear}")
print(f"Minimum Difference (linear after offset): {min_difference_linear}")
print(f"Mean Difference (linear after offset): {mean_difference_linear}")

# Define a custom function to handle phase wrap-around
# Ensures we measure the correct gait phase distance

def calculate_gait_phase_error_sample(ground_truth, prediction):
    if np.abs(ground_truth - prediction) < 0.5:
        return abs(ground_truth - prediction)
    elif ground_truth > prediction:
        return abs(ground_truth - (1 + prediction))
    else:
        return abs((1 + ground_truth) - prediction)

# Vectorize the custom error function
calculate_gait_phase_error = np.vectorize(calculate_gait_phase_error_sample)
filtered_diffs = calculate_gait_phase_error(y_actual, y_pred)

# Compute max, min, mean for the custom error
max_diff = np.max(filtered_diffs)
min_diff = np.min(filtered_diffs)
mean_diff = np.mean(filtered_diffs)

print("------- New Linear Distance with Offset Results -------")
print(f"Maximum Difference (linear after offset): {max_diff}")
print(f"Minimum Difference (linear after offset): {min_diff}")
print(f"Mean Difference (linear after offset): {mean_diff}")

# Visualization of actual vs predicted gait phase
visible_time_range = 500
fig_width_cm = visible_time_range / 50
fig_height = 8

fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height / 2.54

fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

num_points = len(y_phase_original)
pred_start = window_size - 1  # Align predictions with actual data
pred_indices = np.arange(pred_start, pred_start + len(pred_phase))

current_start_time = 0

def update_plot(start_time):
    ax.clear()
    end_time = start_time + visible_time_range
    
    # Ensure we don't go out of bounds
    if end_time > num_points:
        end_time = num_points
    
    # Plot actual data in the visible range
    actual_time_index = np.arange(start_time, end_time)
    actual_data = y_phase_original[start_time:end_time].numpy()
    ax.plot(actual_time_index, actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)

    # Plot predictions if they fall within the range
    pred_mask = (pred_indices >= start_time) & (pred_indices < end_time)
    if np.any(pred_mask):
        visible_pred_indices = pred_indices[pred_mask]
        visible_pred_data = pred_phase[pred_mask]
        ax.plot(visible_pred_indices, visible_pred_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gait Cycle Value', fontsize=12)
    ax.set_title('MLP', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.draw()

# Initial plot
update_plot(current_start_time)

# Create slider to scroll through time
slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(
    slider_ax_time,
    'Time Scroll',
    0,
    max(0, num_points - visible_time_range),
    valinit=0,
    valstep=1
)

def on_time_slider_update(val):
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_time)

slider_time.on_changed(on_time_slider_update)

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])
plt.show()
