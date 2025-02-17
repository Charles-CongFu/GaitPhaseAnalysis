import os
import numpy as np
import pandas as pd
import torch

from typing import Dict, Tuple
from dataclasses import dataclass

from model.lstm import LSTMModel
from train import TrainingConfig
from utils.phase_conversion import phase_to_sincos, sincos_to_phase
from utils.causal_filter import butterworth_filter

@dataclass
class PredictionConfig:
    """Configuration specific to prediction tasks"""
    result_dir: str = None
    test_data_path: str = None
    
    # not necessary, only to get familiar with __post_init__
    predictions_path: str = None
    prediction_times_path: str = None
    prediction_stats_path: str = None
    
    def __post_init__(self):
        # Create result directory
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Initialize prediction-specific paths
        self.predictions_path = os.path.join(self.result_dir, 'predictions.npy')
        self.prediction_times_path = os.path.join(self.result_dir, 'prediction_times.npy')
        self.prediction_stats_path = os.path.join(self.result_dir, 'prediction_stats.txt')

def make_predictions(model: torch.nn.Module, 
                    device: torch.device, 
                    train_config: TrainingConfig,
                    pred_config: PredictionConfig) -> Tuple[np.ndarray, Dict]:
    """
    Make predictions on test data and measure performance metrics
    """
    model.eval()
    
    # Load and preprocess test data
    data = pd.read_csv(pred_config.test_data_path)
    
    # Apply low-pass filter
    for col in train_config.filtered_columns:
        data[col] = butterworth_filter(data[col].values, 
                                     train_config.cutoff_freq, 
                                     train_config.sampling_rate)

    # Prepare features and save ground truth
    features = data[train_config.filtered_columns].values
    true_labels = data['gait_phase'].values
    
    # Convert features to sequences
    sequence_features = []
    for i in range(len(features) - train_config.sequence_length + 1):
        sequence_features.append(features[i:i + train_config.sequence_length])
    sequence_features = np.array(sequence_features)
    sequence_features = torch.FloatTensor(sequence_features)
    
    predictions_list = []
    prediction_times = []
    batch_times = []
    
    with torch.no_grad():
        # Warm up predictions to ensure GPU initialization
        warmup_batch = sequence_features[:train_config.batch_size].to(device)
        for _ in range(10):
            model(warmup_batch)
            
        # Make predictions with timing measurements
        batch_size = 1
        for i in range(0, len(sequence_features), train_config.batch_size):
            batch = sequence_features[i:i + train_config.batch_size].to(device)
        # for i in range(0, len(sequence_features), batch_size):
        #     batch = sequence_features[i:i + batch_size].to(device)
            
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            
            batch_start.record()
            predictions = model(batch)
            phase_pred = sincos_to_phase(predictions.cpu().numpy())
            batch_end.record()
            
            torch.cuda.synchronize()
            batch_time = batch_start.elapsed_time(batch_end)
            batch_times.append(batch_time)
            
            predictions_list.append(phase_pred)
            time_per_prediction = batch_time / len(batch)
            prediction_times.extend([time_per_prediction] * len(batch))
    print(f"batch_times: {np.mean(batch_times)}")
    # Combine predictions
    predictions = np.concatenate(predictions_list)
    
    # Pad initial predictions
    padded_predictions = np.full(len(features), np.nan)
    padded_predictions[train_config.sequence_length-1:] = predictions
    
    # calculate phase space error
    phase_error = np.abs(predictions - true_labels[train_config.sequence_length-1:])
    # deal with wrap-around
    phase_error = np.minimum(phase_error, 1 - phase_error)
    
    prediction_stats = {
        # Timing statistics
        'mean_prediction_time_ms': np.mean(prediction_times),
        'std_prediction_time_ms': np.std(prediction_times),
        'median_prediction_time_ms': np.median(prediction_times),
        'max_prediction_time_ms': np.max(prediction_times),
        'min_prediction_time_ms': np.min(prediction_times),
        'mean_batch_time_ms': np.mean(batch_times),
        'total_predictions': len(predictions),
        
        # Phase space error statistics
        'max_phase_error': np.max(phase_error),
        'min_phase_error': np.min(phase_error),
        'mean_phase_error': np.mean(phase_error),
        'phase_error_std': np.std(phase_error),
    }
    
    # Save results
    np.save(pred_config.predictions_path, padded_predictions)
    np.save(os.path.join(pred_config.result_dir, 'actual_values.npy'), true_labels)
    np.save(pred_config.prediction_times_path, np.array(prediction_times))
    
    return padded_predictions, prediction_stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Model Prediction')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV file')
    args = parser.parse_args()

    # Create configurations
    train_config = TrainingConfig()
    train_config.result_dir = args.model_dir
    
    pred_config = PredictionConfig(
        result_dir=args.model_dir,
        test_data_path=args.test_data
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = LSTMModel(input_size=len(train_config.filtered_columns))
    
    # Load model weights
    try:
        checkpoint = torch.load(os.path.join(train_config.result_dir, 'lstm_model.pth'), 
                              map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print(f"\nLoaded model from {train_config.result_dir}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Make predictions
    print(f"\nMaking predictions on {pred_config.test_data_path}")
    predictions, stats = make_predictions(model, device, train_config, pred_config)
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    
    print("\nPhase Space Error Statistics:")
    print(f"Maximum phase error: {stats['max_phase_error']:.9f}")
    print(f"Minimum phase error: {stats['min_phase_error']:.9f}")
    print(f"Mean phase error: {stats['mean_phase_error']:.9f}")
    print(f"Phase error std: {stats['phase_error_std']:.9f}")
    
    # print("\nTiming Performance:")
    # print(f"Average prediction time: {stats['mean_prediction_time_ms']:.6f} ms")
    # print(f"Maximum prediction time: {stats['max_prediction_time_ms']:.6f} ms")
    # print(f"Batch processing time: {stats['mean_batch_time_ms']:.6f} ms")
    # print(f"Total predictions: {stats['total_predictions']:,}")
    
    # Save detailed statistics
    with open(pred_config.prediction_stats_path, 'w') as f:
        f.write('Test Prediction Statistics:\n')
        f.write('-' * 50 + '\n')
        
        f.write('\nPhase Space Error Statistics:\n')
        f.write(f'Maximum phase error: {stats["max_phase_error"]:.9f}\n')
        f.write(f'Minimum phase error: {stats["min_phase_error"]:.9f}\n')
        f.write(f'Mean phase error: {stats["mean_phase_error"]:.9f}\n')
        f.write(f'Phase error std: {stats["phase_error_std"]:.9f}\n')
        
        f.write('\nTiming Statistics:\n')
        f.write(f'Average prediction time: {stats["mean_prediction_time_ms"]:.6f} ms\n')
        f.write(f'Maximum prediction time: {stats["max_prediction_time_ms"]:.6f} ms\n')
        f.write(f'Batch processing time: {stats["mean_batch_time_ms"]:.6f} ms\n')
        f.write(f'Total predictions: {stats["total_predictions"]}\n')

if __name__ == "__main__":
    main()