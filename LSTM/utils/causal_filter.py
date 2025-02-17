import numpy as np
from scipy.signal import butter, lfilter

def butterworth_filter(data: np.ndarray, cutoff: float, fs: float, 
                      order: int = 4) -> np.ndarray:
    """
    Butterworth lowpass causal filter
    
    Args:
        data: input data
        cutoff: cutoff frequency
        fs: sampling frequency
        order: filter order
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data) # causal filter, forward only
    return filtered_data