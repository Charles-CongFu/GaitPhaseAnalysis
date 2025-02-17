import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Low-pass filter function
def low_pass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# High-pass filter function (for removing integration drift)
def high_pass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Read data
file_path = './motion_data_with_ground_truth_-0.2_45.csv'
data = pd.read_csv(file_path)

# Sampling frequency
fs = 100
cutoff_freq = 3.0

# Integrate angular velocity to obtain hip angle
data['hip_angle'] = np.cumsum(data['imu_ang_vel_z'])

# Normalize angle (set starting point to 0)
data['hip_angle'] -= data['hip_angle'][0]

# Apply low-pass filtering
data['hip_angle_filtered'] = low_pass_filter(data['hip_angle'], cutoff_freq, fs)
data['hip_angular_velocity_filtered'] = low_pass_filter(data['imu_ang_vel_z'], cutoff_freq, fs)

# Compute polar coordinates
phi_raw = np.arctan2(data['hip_angular_velocity_filtered'], data['hip_angle_filtered'])
phi_corrected = np.abs((phi_raw + 2 * np.pi) % (2 * np.pi))

heel_strike_indices = []
heel_strike_angles = []

phi_HS = phi_corrected.iloc[0]  # Use the first gait cycle's Heel Strike angle

for t in range(1, len(data) - 1):  # Avoid index out of range errors
    # **Heel Strike detection based on φ_t crossing zero**
    if (phi_corrected.iloc[t] > 6.0) and (phi_corrected.iloc[t + 1] < 0.1):
        if len(heel_strike_indices) == 0 or (t - heel_strike_indices[-1] > fs * 0.5):  # At least 0.5s interval
            phi_HS = phi_corrected.iloc[t]  # Record new gait cycle start
            heel_strike_indices.append(t)
            heel_strike_angles.append(phi_HS)

    # **Compute gait phase**
    data.loc[t, 'phi_t'] = (phi_corrected.iloc[t] - phi_HS) % (2 * np.pi)

    # **Compute predicted gait phase**
    data['predicted_phase'] = np.abs((data['phi_t'] - 2 * np.pi) / (2 * np.pi)) % 1

    # **Compute ground truth gait phase**
    data['groundtruth_phase'] = (data['gait_phase'] - data['gait_phase'].min()) / (
            data['gait_phase'].max() - data['gait_phase'].min())

# Visualization of φ_t
plt.figure(figsize=(12, 6))
plt.plot(data['phi_t'][:500], color='blue', linestyle='-', linewidth=1.5, label=r"$\phi_t$ (Current Phase Angle)")
plt.xlabel("Time Step")
plt.ylabel(r"$\phi_t$ (radians)")
plt.title("Current Phase Angle Over Time (Fixed)")
plt.legend()
plt.grid()
plt.show()

# **Compare Predicted Phase and Ground Truth Phase**
num_samples = 500
plt.figure(figsize=(12, 6))
plt.plot(np.arange(num_samples), data['predicted_phase'][:num_samples], label="Predicted Gait Phase",
         color='red', linestyle='--', linewidth=1.5)
plt.plot(np.arange(num_samples), data['groundtruth_phase'][:num_samples], label="Groundtruth",
         color='blue', linestyle='-', linewidth=1.5)

plt.xlabel("Time Step (First 2000)")
plt.ylabel("Gait Phase")
plt.title("Comparison of Predicted and Groundtruth Gait Phases")
plt.legend()
plt.grid()
plt.show()

# **Visualization of integrated and filtered hip angle**
num_samples = 500

plt.figure(figsize=(12, 6))
plt.plot(data['hip_angle'][:num_samples], color='purple', linestyle='-', linewidth=1.5, label="Hip Angle (Integrated)")
plt.plot(data['hip_angle_filtered'][:num_samples], color='blue', linestyle='--', linewidth=1.5, label="Hip Angle (Filtered)")
plt.xlabel("Time Step (First 2000)")
plt.ylabel("Hip Angle (degrees)")
plt.title("Integrated and Filtered Hip Angle Over Time (First 2000 Samples)")
plt.legend()
plt.grid()
plt.show()
