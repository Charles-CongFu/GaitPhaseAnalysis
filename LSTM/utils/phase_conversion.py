import numpy as np



def phase_to_sincos(phase):
    """Convert phase (0~1) to sin/cos representation"""
    theta = 2 * np.pi * phase
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)

def sincos_to_phase(sin_cos):
    """Convert sin/cos representation back to phase (0~1)"""
    phase = np.arctan2(sin_cos[..., 0], sin_cos[..., 1]) / (2 * np.pi)
    return (phase + 1) % 1  # wrap to [0, 1]