# Standard library imports
import json
import os
import time

# Third-party imports
import pytorch_lightning as pl
import torch

class BestMetricsCallback(pl.Callback):
    def __init__(self, metrics_to_track, save_path=None):
        super().__init__()
        self.metrics_to_track = metrics_to_track
        self.save_path = save_path
        self.best_metrics = {metric: None for metric in metrics_to_track}
        self.best_epochs = {metric: None for metric in metrics_to_track}
        
        self.train_start_time = None
        self.train_end_time = None
        self.training_time = None
        self.model_size = None
        self.inference_rate = None

    def on_fit_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        
        # Compute model size
        self.model_size = self.compute_model_size(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_epoch = trainer.current_epoch

        for metric in self.metrics_to_track:
            current = logs.get(metric)

            if current is None:
                continue  

            if self.best_metrics[metric] is None:
                self.best_metrics[metric] = current
                self.best_epochs[metric] = current_epoch
                continue

            if self.is_metric_better(metric, current, self.best_metrics[metric]):
                self.best_metrics[metric] = current
                self.best_epochs[metric] = current_epoch

    def on_test_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_epoch = trainer.current_epoch

        for metric in self.metrics_to_track:
            current = logs.get(metric)

            if current is None:
                continue  

            if self.best_metrics[metric] is None:
                self.best_metrics[metric] = current
                self.best_epochs[metric] = current_epoch
                continue

            if self.is_metric_better(metric, current, self.best_metrics[metric]):
                self.best_metrics[metric] = current
                self.best_epochs[metric] = current_epoch

    def on_train_end(self, trainer, pl_module):
        self.train_end_time = time.time()
        self.training_time = self.train_end_time - self.train_start_time  

        # Compute inference rate
        self.inference_rate = self.compute_inference_rate(pl_module, trainer, pl_module.device)

        # Convert tensors to Python scalars for JSON serialization
        best_metrics_python = {}
        for metric, value in self.best_metrics.items():
            if isinstance(value, torch.Tensor):
                best_metrics_python[metric] = value.item()
            else:
                best_metrics_python[metric] = value

        best_epochs_python = {}
        for metric, epoch in self.best_epochs.items():
            best_epochs_python[metric] = epoch

        # Convert training time to hours and minutes
        hours, rem = divmod(self.training_time, 3600)
        minutes, _ = divmod(rem, 60)
        training_time_formatted = f"{int(hours)}h {int(minutes)}m"

        # Prepare the data to save
        data_to_save = {
            'best_metrics': best_metrics_python,
            'best_epochs': best_epochs_python,
            'training_time_sec': self.training_time,
            'training_time_formatted': training_time_formatted,
            'model_size_MB': self.model_size,
            'inference_rate_images_per_sec': self.inference_rate
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Save to JSON
        with open(self.save_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        # Print the saved metrics
        print(f"\nBest Metrics saved to {self.save_path}:")
        for metric in self.metrics_to_track:
            value = best_metrics_python.get(metric, 'N/A')
            epoch = best_epochs_python.get(metric, 'N/A')
            print(f"  {metric}: {value} (Epoch {epoch})")
        
        # Print additional metrics
        print(f"  Training Time: {self.training_time:.2f} seconds")
        print(f"  Model Size: {self.model_size:.2f} MB")
        print(f"  Inference Rate: {self.inference_rate:.2f} images/second")

    def on_test_end(self, trainer, pl_module):
        # Convert tensors to Python scalars for JSON serialization
        best_metrics_python = {}
        for metric, value in self.best_metrics.items():
            if isinstance(value, torch.Tensor):
                best_metrics_python[metric] = value.item()
            else:
                best_metrics_python[metric] = value

        best_epochs_python = {}
        for metric, epoch in self.best_epochs.items():
            best_epochs_python[metric] = epoch

        # Prepare the data to save
        data_to_save = {
            'best_metrics': best_metrics_python,
            'best_epochs': best_epochs_python,
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Save to JSON
        with open(self.save_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        # Print the saved metrics
        print(f"\nBest Metrics saved to {self.save_path}:")
        for metric in self.metrics_to_track:
            value = best_metrics_python.get(metric, 'N/A')
            epoch = best_epochs_python.get(metric, 'N/A')
            print(f"  {metric}: {value} (Epoch {epoch})")
        
    def is_metric_better(self, metric, current, best):
        metrics_to_maximize = ['val_acc', 'val_f1', 'val_precision', 'val_recall', 'val_f2', 'val_avg_precision', 'test_acc', 'test_f1', 'test_precision', 'test_recall', 'test_f2', 'test_avg_precision']
        metrics_to_minimize = ['val_loss', 'val_one_error', 'val_hamming_loss', 'test_loss', 'test_one_error', 'test_hamming_loss']

        if metric in metrics_to_maximize:
            return current > best
        elif metric in metrics_to_minimize:
            return current < best
        else:
            # Default to maximize if not specified
            return current > best

    def compute_model_size(self, pl_module):
        total_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 ** 2)  
        return model_size_mb

    def compute_inference_rate(self, pl_module, trainer, device):
        test_dataloader = trainer.datamodule.test_dataloader()
        try:
            batch = next(iter(test_dataloader))
        except StopIteration:
            print("Test dataloader is empty. Cannot compute inference rate.")
            return None

        x, y = batch
        x = x.to(device)

        pl_module.eval()
        with torch.no_grad():
            # Warm-up iterations
            for _ in range(5):
                pl_module(x)
            if 'cuda' in device.type:
                torch.cuda.synchronize()

            # Time the inference
            start_time = time.time()
            pl_module(x)
            # Synchronize to ensure the forward pass completes
            if 'cuda' in device.type:
                torch.cuda.synchronize()
            end_time = time.time()

        inference_time = end_time - start_time
        if inference_time > 0:
            inference_rate = len(x) / inference_time
        else:
            inference_rate = float('inf')  

        return inference_rate
