import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dataclasses import dataclass
from typing import Optional

@dataclass
class PlotConfig:
    """Configuration for plotting gait cycle data"""
    result_dir: str
    visible_time_range: int = 500  # how many time steps to show at once
    fig_height: float = 12.0  # figure height in inches
    
    # File paths will be automatically set based on result_dir
    predictions_path: Optional[str] = None
    actual_values_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize file paths based on result directory"""
        self.predictions_path = os.path.join(self.result_dir, 'predictions.npy')
        self.actual_values_path = os.path.join(self.result_dir, 'actual_values.npy')
        
        # Verify files exist
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")
        if not os.path.exists(self.actual_values_path):
            raise FileNotFoundError(f"Actual values file not found: {self.actual_values_path}")

def plot_gait_cycle(config: PlotConfig):
    """Plot gait cycle predictions with interactive time slider
    
    Args:
        config: PlotConfig object containing plotting parameters and file paths
    """
    # Load pre-computed data
    actual_labels = np.load(config.actual_values_path)
    predictions = np.load(config.predictions_path)
    
    # Calculate phase difference considering periodicity
    raw_difference = actual_labels - predictions
    # deal with periodic gait cycles
    phase_difference = np.where(raw_difference > 0.5, raw_difference - 1,
                              np.where(raw_difference < -0.5, raw_difference + 1, raw_difference))
    
    # lower and upper bound of plot
    y_min, y_max = -0.5, 0.5
    
    # Plotting setup
    fig_width_cm = config.visible_time_range / 50
    fig = plt.figure(figsize=(fig_width_cm / 2.54, config.fig_height / 2.54))
    
    # Create main axes for actual vs predicted
    main_ax = fig.add_axes([0.1, 0.6, 0.8, 0.35])
    
    # Create axes for the difference (error)
    diff_ax = fig.add_axes([0.1, 0.15, 0.8, 0.35])

    def update_plot(start_time):
        """Update plot based on slider value
        
        Args:
            start_time: Starting time index for the plot window
        """
        main_ax.clear()
        diff_ax.clear()
        end_time = start_time + config.visible_time_range
        
        # Get data slice for current view
        actual_data = actual_labels[int(start_time):int(end_time)]
        predicted_data = predictions[int(start_time):int(end_time)]
        difference = phase_difference[int(start_time):int(end_time)]
        
        # Plot actual vs predicted
        main_ax.plot(actual_data, label='Actual', color='blue', linewidth=1.5)
        main_ax.plot(predicted_data, label='Predicted', color='red', linewidth=1.5)
        main_ax.set_xlabel('Time Step', fontsize=12)
        main_ax.set_ylabel('Gait Cycle Value', fontsize=12)
        main_ax.set_title('Actual vs Predicted Gait Cycle', fontsize=16)
        main_ax.legend(loc='upper right', fontsize=10)
        main_ax.grid(True, linestyle='--', alpha=0.5)
        main_ax.set_ylim(-0.1, 1.1) # extra margin for plot 

        # Plot difference
        diff_ax.plot(difference, label='Circular Difference', 
                    color='green', linewidth=1.5)
        diff_ax.set_ylim(y_min, y_max)  # ±0.5
        diff_ax.set_xlabel('Time Step', fontsize=12)
        diff_ax.set_ylabel('Phase Difference', fontsize=12)
        diff_ax.set_title('Circular Phase Difference Over Time', fontsize=16)
        diff_ax.legend(loc='upper right', fontsize=10)
        diff_ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.draw()

    # Add slider for time navigation
    slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], 
                            facecolor='lightgoldenrodyellow')
    slider_time = Slider(
        slider_ax_time,
        'Time Scroll',
        0,
        max(0, len(actual_labels) - config.visible_time_range),
        valinit=0,
        valstep=1
    )
    
    # Bind slider to update function
    slider_time.on_changed(lambda val: update_plot(val))
    
    # Initial plot
    update_plot(0)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot Gait Cycle Predictions')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory containing prediction results')
    parser.add_argument('--visible_range', type=int, default=500,
                       help='Number of time steps to show at once')
    args = parser.parse_args()

    # Create plot configuration
    config = PlotConfig(
        result_dir=args.result_dir,
        visible_time_range=args.visible_range
    )
    
    # Generate plot
    plot_gait_cycle(config)

if __name__ == "__main__":
    main()