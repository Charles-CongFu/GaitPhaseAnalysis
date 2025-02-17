import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

seed = 42
window_size = 30

# ========== seed ==========
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

class CNNModel(nn.Module):
    def __init__(self, in_channels=6, window_size=30, out_dim=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(32 * window_size, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def calculate_gait_phase_error_sample(ground_truth, prediction):

    diff = abs(ground_truth - prediction)
    if diff < 0.5:
        return diff
    else:
        return 1.0 - diff

calculate_gait_phase_error = np.vectorize(calculate_gait_phase_error_sample)


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
    "motion_data_with_ground_truth_-0.35_45.csv",

]

all_dataset_labels = []
all_max_errors = []
all_min_errors = []
all_mean_errors = []

for file_name in file_names:
    file_path = os.path.join(file_dir, file_name)
    data = pd.read_csv(file_path)

    file_name_wo_ext = os.path.splitext(file_name)[0]
    print(f"\n===== Test on dataset: {file_name_wo_ext} =====")


    dataset_label = file_name_wo_ext.replace("motion_data_with_ground_truth_", "")

    # data['imu_acc_x'] = data['imu_acc_x'].clip(-10, 5)
    # data['imu_acc_y'] = data['imu_acc_y'].clip(-10, 5)
    # data['imu_ang_acc_z'] = data['imu_ang_acc_z'].clip(-40, 20)

    cutoff_freq = 5.0
    fs = 100
    filtered_columns = [
        'imu_acc_x', 'imu_acc_y', 'imu_acc_z',
        'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z'
    ]
    for col in filtered_columns:
        if col in data.columns:
            data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

    features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
    label = 'gait_phase'

    X = data[features].values
    y = data[label].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_min = X_tensor.min(dim=0, keepdim=True).values
    X_max = X_tensor.max(dim=0, keepdim=True).values
    X_tensor = (X_tensor - X_min) / (X_max - X_min)
    X = X_tensor.numpy()

    y_phase_original = torch.tensor(y, dtype=torch.float32)
    angle = 2 * math.pi * y_phase_original
    y_cos = torch.cos(angle)
    y_sin = torch.sin(angle)
    y_encoded = torch.stack([y_cos, y_sin], dim=-1).numpy()

    X_windows = []
    y_windows = []
    for i in range(len(X) - window_size + 1):
        window_data = X[i: i + window_size]
        window_data = window_data.T
        label_data = y_encoded[i + window_size - 1]
        X_windows.append(window_data)
        y_windows.append(label_data)

    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.float32)

    X_test_tensor = torch.from_numpy(X_windows)
    y_test_tensor = torch.from_numpy(y_windows)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    MODEL_DIR = "./CNN/model"
    model_filename = "all_positions_cnn_window_model.pth"  # 你要加载的模型
    model_load_path = os.path.join(MODEL_DIR, model_filename)  # 生成完整路径
    if not os.path.exists(model_load_path):
        print(f"Error: model file not found: {model_load_path}")
        continue

    model = CNNModel(in_channels=len(features), window_size=window_size, out_dim=2).to(device)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_load_path}")

    all_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)  # [batch_size, 2]
            all_preds.append(outputs.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)  # [num_windows, 2]


    pred_angle = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    pred_phase = (pred_angle / (2 * math.pi)) % 1.0

    y_phase = y_phase_original[window_size - 1:].numpy()

    # ========== (corrected_error) ==========
    corrected_error = calculate_gait_phase_error(y_phase, pred_phase)
    max_err = np.max(corrected_error)
    min_err = np.min(corrected_error)
    mean_err = np.mean(corrected_error)

    print(f"--- Test Results on {file_name_wo_ext} ---")
    print(f"Max Circular Error: {max_err}")
    print(f"Min Circular Error: {min_err}")
    print(f"Mean Circular Error: {mean_err}")

    all_dataset_labels.append(dataset_label)
    all_max_errors.append(max_err)
    all_min_errors.append(min_err)
    all_mean_errors.append(mean_err)

    best_offset = None
    best_mean_diff = float('inf')
    for offset in np.linspace(0, 1, 101):
        y_pred_shifted = (pred_phase + offset) % 1.0
        shifted_err = calculate_gait_phase_error(y_phase, y_pred_shifted)
        mean_diff = np.mean(shifted_err)
        if mean_diff < best_mean_diff:
            best_mean_diff = mean_diff
            best_offset = offset
    y_pred_aligned = (pred_phase + best_offset) % 1.0

    final_err = calculate_gait_phase_error(y_phase, y_pred_aligned)
    print(f"Best offset: {best_offset:.2f}")
    print(f"Mean Error after offset: {final_err.mean():.4f}")


    visible_time_range = 2000
    fig_width_cm = visible_time_range / 10
    fig_height = 8

    fig_width_inches = fig_width_cm / 2.54
    fig_height_inches = fig_height / 2.54

    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
    #fig, ax = plt.subplots(figsize=(12, 6))
    current_start_time = 0

    def update_plot(start_idx):
        ax.clear()
        end_idx = start_idx + visible_time_range
        actual_data = y_phase[int(start_idx):int(end_idx)]
        predicted_data = y_pred_aligned[int(start_idx):int(end_idx)]

        ax.plot(actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)
        ax.plot(predicted_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)

        ax.set_xlabel('Window Index', fontsize=12)
        ax.set_ylabel('Gait Cycle Value', fontsize=12)
        ax.set_title(f'Actual vs Predicted', fontsize=16)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.draw()

    update_plot(current_start_time)

    slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
    slider_time = Slider(
        slider_ax_time,
        'Window Scroll',
        0,
        max(0, len(y_phase) - visible_time_range),
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


if all_dataset_labels:
    fig_summary, axs = plt.subplots(4, 1, figsize=(12, 16))  # 改成 4 行子图

    # Max Error
    axs[0].plot(all_dataset_labels, all_max_errors, marker='o', linestyle='-', color='r', label='Max Error')
    axs[0].set_title("Max Corrected Error")
    axs[0].set_ylabel("Corrected Error")
    axs[0].tick_params(axis='x', labelrotation=45)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # Min Error
    axs[1].plot(all_dataset_labels, all_min_errors, marker='s', linestyle='-', color='g', label='Min Error')
    axs[1].set_title("Min Corrected Error")
    axs[1].set_ylabel("Corrected Error")
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    # Mean Error
    axs[2].plot(all_dataset_labels, all_mean_errors, marker='d', linestyle='-', color='b', label='Mean Error')
    axs[2].set_title("Mean Corrected Error")
    axs[2].set_ylabel("Corrected Error")
    axs[2].tick_params(axis='x', labelrotation=45)
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend()

    # Max/Min/Mean
    axs[3].plot(all_dataset_labels, all_max_errors, marker='o', linestyle='-', color='r', label='Max Error')
    axs[3].plot(all_dataset_labels, all_min_errors, marker='s', linestyle='-', color='g', label='Min Error')
    axs[3].plot(all_dataset_labels, all_mean_errors, marker='d', linestyle='-', color='b', label='Mean Error')
    axs[3].set_title("Max / Min / Mean Corrected Error Comparison")
    axs[3].set_ylabel("Corrected Error")
    axs[3].tick_params(axis='x', labelrotation=45)
    axs[3].grid(True, linestyle="--", alpha=0.5)
    axs[3].legend()

    plt.tight_layout()

    plt.show()