import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.signal import butter, lfilter

# ========== 超参数区域 ==========
seed = 42
window_size = 30
batch_size = 256
num_epochs = 100
learning_rate = 1e-3

# ========== 设置随机种子、设备等 ==========
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== 低通滤波器函数 ==========
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# ========== 定义 CNN 模型 ==========
class CNNModel(nn.Module):
    def __init__(self, in_channels=6, window_size=30, out_dim=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        # 这里可以加池化或更多层
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

# ========== 早停类 ==========
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
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

# ========== 需要训练的文件列表 (示例) ==========
file_dir =  './'
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

# ========= 准备空列表收集所有文件的数据 =========
all_X_windows = []
all_y_windows = []

# ========= 定义好需要滤波和提取的列 =========
filtered_columns = [
    'imu_acc_x', 'imu_acc_y', 'imu_acc_z',
    'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z'
]
cutoff_freq = 5.0
fs = 100
label = 'gait_phase'

for file_name in file_names:
    file_path = os.path.join(file_dir, file_name)
    data = pd.read_csv(file_path)

    # 低通滤波
    for col in filtered_columns:
        if col in data.columns:
            data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)


    features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]

    X = data[features].values  # shape: [N, num_features]
    y = data[label].values     # shape: [N]

    # normal（0-1）
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_min = X_tensor.min(dim=0, keepdim=True).values
    X_max = X_tensor.max(dim=0, keepdim=True).values
    X_normed = (X_tensor - X_min) / (X_max - X_min)
    X = X_normed.numpy()

    # gait_phase -> cos,sin
    y_phase_original = torch.tensor(y, dtype=torch.float32)
    angle = 2 * math.pi * y_phase_original
    y_cos = torch.cos(angle)
    y_sin = torch.sin(angle)
    y_encoded = torch.stack([y_cos, y_sin], dim=-1).numpy()

    # ========= 滑窗 =========
    num_samples = len(X)
    for i in range(num_samples - window_size + 1):
        # 取window_size长度的数据
        window_data = X[i:i+window_size]    # [window_size, num_features]
        window_data = window_data.T         # 变为 [num_features, window_size]
        label_data = y_encoded[i + window_size - 1]  # [2]
        all_X_windows.append(window_data)
        all_y_windows.append(label_data)

all_X_windows = np.array(all_X_windows, dtype=np.float32)  # [total_num_windows, in_channels, window_size]
all_y_windows = np.array(all_y_windows, dtype=np.float32)  # [total_num_windows, 2]

X_tensor = torch.from_numpy(all_X_windows)  # shape: [N, channels, window]
y_tensor = torch.from_numpy(all_y_windows)  # shape: [N, 2]
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = CNNModel(
    in_channels=len(filtered_columns),  # 这里要与 features 对应的通道数一致
    window_size=window_size,
    out_dim=2
).to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-3)
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

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
    early_stopping(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

model_output_dir = "./CNN/model"
os.makedirs(model_output_dir, exist_ok=True)
model_save_path = os.path.join(model_output_dir, "all_positions_cnn_window_model.pth")

torch.save(model.state_dict(), model_save_path)
print(f"Unified model saved to {model_save_path}")

print("\nTraining for all files done!")
