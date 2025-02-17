import numpy as np
import random

from typing import List, Callable, Dict

# Augmentation for sim2real
class IMUAugmentor:
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # define augmentation methods
        self.augmentation_methods: Dict[str, Callable] = {
            'gaussian_noise': self.add_gaussian_noise,
            'random_bias': self.add_random_bias,
            'magnitude_scaling': self.apply_magnitude_scaling,
            'sensor_drift': self.add_sensor_drift,
            'spike_noise': self.add_spike_noise
        }
    
    def add_gaussian_noise(self, data: np.ndarray, std_range: tuple = (0.01, 0.1)) -> np.ndarray:
        std = np.random.uniform(*std_range)
        noise = np.random.normal(0, std, size=data.shape)
        return data + noise
    
    def add_random_bias(self, data: np.ndarray, bias_range: tuple = (-0.1, 0.1)) -> np.ndarray:
        bias = np.random.uniform(*bias_range, size=data.shape[1])
        return data + bias
    
    def apply_magnitude_scaling(self, data: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
        scale = np.random.uniform(*scale_range, size=data.shape[1])
        return data * scale
    
    def add_sensor_drift(self, data: np.ndarray, drift_scale: float = 0.001) -> np.ndarray:
        t = np.arange(len(data))
        drift = drift_scale * np.random.randn(data.shape[1]) * t[:, np.newaxis]
        return data + drift
    
    # spike noise to simulate instantaneous interference
    def add_spike_noise(self, data: np.ndarray, 
                       probability: float = 0.01, 
                       amplitude_range: tuple = (-2, 2)) -> np.ndarray:
        mask = np.random.random(data.shape) < probability
        spikes = np.random.uniform(*amplitude_range, size=data.shape)
        return data + (mask * spikes)
    
    def augment_data(self, 
                    data: np.ndarray,
                    num_augmentations: int = 1,
                    methods: List[str] = None,
                    probabilities: List[float] = None) -> np.ndarray:
        """
        augment input data
        
        Args:
            data: size (n_samples, n_features)
            num_augmentations: number of augmented samples to generate
            methods: augmentation methods to apply, if None, all methods are used
            probabilities: probability of selecting each method, must sum to 1, if None, equal probability is used
            
        Returns:
            augmented_data: size (n_samples * num_augmentations, n_features)
        """
        if methods is None:
            methods = list(self.augmentation_methods.keys())
        
        if probabilities is None:
            probabilities = [1.0 / len(methods)] * len(methods)
        
        # normalize probabilities
        probabilities = np.array(probabilities) / sum(probabilities)
        
        # copy original data
        augmented_data = data.copy()
        
        # randomly select augmentation methods
        selected_methods = np.random.choice(
            methods, 
            size=num_augmentations, 
            p=probabilities,
            replace=False
        )
        
        for method_name in selected_methods:
            augmented_data = self.augmentation_methods[method_name](augmented_data)
            
        return augmented_data