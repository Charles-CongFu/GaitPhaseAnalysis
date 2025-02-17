import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import butter, filtfilt, lfilter

# Low-pass filter function
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
    通用 Butterworth 滤波器函数

    参数:
        - data: 待滤波的数据 (1D array)
        - cutoff: 截止频率，单值（低通/高通）或元组（带通/带阻）
        - fs: 采样频率 (Hz)
        - order: 滤波器阶数 (int, 默认 4)
        - filter_type: 滤波类型 ('low', 'high', 'bandpass', 'bandstop')

    返回:
        - 滤波后的数据 (1D array)
    """
    # 归一化截止频率
    nyquist = 0.5 * fs  # 奈奎斯特频率
    if isinstance(cutoff, (list, tuple)):  # 带通或带阻滤波器
        normal_cutoff = [c / nyquist for c in cutoff]
    else:  # 低通或高通滤波器
        normal_cutoff = cutoff / nyquist

    # 设计滤波器
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    
    # 滤波处理
    filtered_data = lfilter(b, a, data)
    return filtered_data

# Read the CSV file
file_path = './data/motion_data_with_ground_truth_0.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['imu_acc_x'] = data['imu_acc_x'].clip(-25, 40)
data['imu_acc_y'] = data['imu_acc_y'].clip(-20, 40)
data['imu_ang_vel_z'] = data['imu_ang_vel_z'].clip(-3, 4)

# Select the columns to plot
selected_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
columns_to_plot = []

# Add original data, low-pass filtered data, and gait_cycle
cutoff_freq = 5.0  # Cutoff frequency (Hz)
fs = 100  # Sampling frequency (Hz)

for col in selected_columns:
    columns_to_plot.append((col, "original"))  # Original data
    filtered_col = f"{col}_filtered"
    if col in data.columns:
        data[filtered_col] = butterworth_filter(data[col], cutoff_freq, fs)  # Add low-pass filtered column
        columns_to_plot.append((filtered_col, "filtered"))  # Filtered data
    if 'gait_cycle' in data.columns:  # Insert gait_cycle
        columns_to_plot.append(('gait_cycle', "gait_cycle"))

# If no columns are specified, default to plotting all columns
if not columns_to_plot:
    columns_to_plot = [(col, "original") for col in data.columns[1:]]

visible_rows = 4  # Number of rows displayed at once
visible_time_range = 10  # Initial horizontal display time range (in seconds)


fig_width = 5  # Figure width
fig_height = visible_rows * 3  # Each subplot is 3 units high

# Create the figure and subplots
fig, ax = plt.subplots(visible_rows, 1, figsize=(fig_width, fig_height), sharex=True)
current_start_col = 0  # Current starting column index
current_start_time = 0  # Current starting time

def update_plot(start_col, start_time):
    """Update the plot"""
    start_col = int(start_col)
    for i in range(visible_rows):
        ax[i].clear()
        if start_col + i < len(columns_to_plot):
            column, col_type = columns_to_plot[start_col + i]
            # Filter data within the horizontal time range
            mask = (data['time'] >= start_time) & (data['time'] <= start_time + visible_time_range)
            if col_type == "gait_cycle":  # If gait_cycle, plot with a red line
                ax[i].plot(data['time'][mask], data[column][mask], label=column, color='red')
            elif col_type == "filtered":  # If low-pass filtered data, plot with a green line
                ax[i].plot(data['time'][mask], data[column][mask], label=column, color='green')
            else:  # Plot original data normally
                ax[i].plot(data['time'][mask], data[column][mask], label=column)
            ax[i].set_ylabel(column, fontsize=10)
            ax[i].legend(loc='upper right', fontsize=8)
            ax[i].grid(True)
        else:
            ax[i].axis('off')  # Hide extra subplots

    ax[-1].set_xlabel('Time (s)', fontsize=12)  # Set time as the x-axis
    plt.draw()

update_plot(current_start_col, current_start_time)

# Add vertical slider for scrolling through columns
slider_ax_col = plt.axes([0.01, 0.2, 0.02, 0.6], facecolor='lightgrey')
slider_col = Slider(slider_ax_col, 'Column Scroll', 0, max(0, len(columns_to_plot) - visible_rows), valinit=0, valstep=1, orientation='vertical')

# Add horizontal slider for scrolling through time
slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(slider_ax_time, 'Time Scroll', 0, max(0, data['time'].max() - visible_time_range), valinit=0, valstep=0.1)

# Bind slider events
def on_col_slider_update(val):
    global current_start_col
    current_start_col = int(slider_col.val)
    update_plot(current_start_col, current_start_time)

def on_time_slider_update(val):
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_col, current_start_time)

slider_col.on_changed(on_col_slider_update)
slider_time.on_changed(on_time_slider_update)

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])  # Leave space for sliders
plt.show()
