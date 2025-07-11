# Standard library imports
import json
import os
import time

# Third-party imports
import pytorch_lightning as pl
import torch

# Callback to save the best metrics to a JSON file
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
    
    # Save temporary metrics to a JSON file
    def save_temp_metrics(self):
        temp_save_path = self.save_path.replace('best_metrics.json', 'best_metrics_temp.json')
        best_metrics_python = {metric: (value.item() if isinstance(value, torch.Tensor) else value)
                               for metric, value in self.best_metrics.items()}
        best_epochs_python = {metric: epoch for metric, epoch in self.best_epochs.items()}

        data_to_save = {
            'best_metrics': best_metrics_python,
            'best_epochs': best_epochs_python,
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)
        with open(temp_save_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    
    # Record training start time and compute model size
    def on_fit_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.model_size = self.compute_model_size(pl_module) 

    # Update best metrics and save them at the end of each validation epoch
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

        self.save_temp_metrics()

    # Update best metrics at the end of the test phase
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

    # Save the best metrics and training details at the end of training
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

    # Save the best metrics at the end of the test phase
    def on_test_end(self, trainer, pl_module):
        best_metrics_python = {} 

        # Convert tensors to Python scalars for JSON serialization
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

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True) 
        with open(self.save_path, 'w') as f: 
            json.dump(data_to_save, f, indent=4)

        # Print the saved metrics
        print(f"\nBest Metrics saved to {self.save_path}:")
        for metric in self.metrics_to_track:
            value = best_metrics_python.get(metric, 'N/A')
            epoch = best_epochs_python.get(metric, 'N/A')
            print(f"  {metric}: {value} (Epoch {epoch})")
    
    # Check if the current metric is better than the best recorded metric
    def is_metric_better(self, metric, current, best):
        metrics_to_maximize = ['val_acc', 'val_f1', 'val_precision', 'val_recall', 'val_f2', 'val_avg_precision', 
                               'test_acc', 'test_f1', 'test_precision', 'test_recall', 'test_f2', 'test_avg_precision']
        metrics_to_minimize = ['val_loss', 'val_one_error', 'val_hamming_loss', 'test_loss', 'test_one_error', 'test_hamming_loss']

        if metric in metrics_to_maximize:
            return current > best
        elif metric in metrics_to_minimize:
            return current < best
        else:
            # Default to maximize if not specified
            return current > best

    # Compute the model size in MB
    def compute_model_size(self, pl_module):
        total_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 ** 2)  
        return model_size_mb

    # Compute the inference rate in images per second
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

# Callback to log the start and end of each epoch
class LogEpochEndCallback(pl.Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_epoch_start(self, trainer, pl_module):
        self.logger.info(f"Starting training epoch {trainer.current_epoch}.")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.logger.info(f"Starting validation epoch {trainer.current_epoch}.")
    
    def on_test_epoch_start(self, trainer, pl_module):
        self.logger.info(f"Starting test epoch {trainer.current_epoch}.")
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.logger.info(f"Training epoch {trainer.current_epoch} finished.")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.logger.info(f"Validation epoch {trainer.current_epoch} finished.")

    def on_test_epoch_end(self, trainer, pl_module):
        self.logger.info(f"Test epoch {trainer.current_epoch} finished.")

# Callback to log the gradient norm after each backward pass
class GradientLoggingCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):

        # Log gradient norm before clipping
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        pl_module.log("gradient_norm_before_clipping", total_norm, on_step=True, on_epoch=False)
        print(f"Gradient Norm before clipping: {total_norm:.4f}")

    # Log gradient norm after clipping and before optimizer step
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total_norm = 0.0 # Log gradient norm after clipping but before optimizer scaling
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        pl_module.log("gradient_norm_after_clipping_raw", total_norm, on_step=True, on_epoch=False)
        print(f"Gradient Norm after clipping (raw): {total_norm:.4f}")

        # Log effective norm after learning rate scaling
        lr = optimizer.param_groups[0]['lr']  
        effective_norm = total_norm * lr
        pl_module.log("gradient_norm_effective", effective_norm, on_step=True, on_epoch=False)
        print(f"Gradient Norm after clipping (effective, LR={lr}): {effective_norm:.4f}")

# Callback to log the learning rate when it changes
class OnChangeLrLoggerCallback(pl.Callback):
    def __init__(self, custom_logger):
        self.custom_logger = custom_logger
        self.prev_lr = None

    # Log the learning rate when it changes
    def on_train_epoch_end(self, trainer, pl_module):
        optimizer = trainer.optimizers[0] 
        current_lr = optimizer.param_groups[0]['lr']

        # Log for the first epoch or if the learning rate has changed
        if self.prev_lr is None:
            self.custom_logger.info(f"Initial Learning Rate: {current_lr}")
        elif self.prev_lr != current_lr:
            self.custom_logger.info(f"Epoch {trainer.current_epoch}: Learning Rate changed to: {current_lr}")

        self.prev_lr = current_lr # Update previous learning rate for next epoch comparisons
        
# Callback to log when early stopping occurs
class EarlyStoppingLoggerCallback(pl.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.stopped_epoch = 0

    # Log when early stopping is triggered
    def on_validation_end(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.EarlyStopping) and callback.stopped_epoch > 0:
                if self.stopped_epoch == 0:  # Log only once
                    self.stopped_epoch = callback.stopped_epoch
                    val_loss = trainer.callback_metrics.get('val_loss', 'N/A')
                    self.logger.info(f"Early stopping triggered at epoch {self.stopped_epoch} with val_loss: {val_loss:.4f}")