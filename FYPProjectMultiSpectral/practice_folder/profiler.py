import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch will use the CPU.")


num_gpus = torch.cuda.device_count()  # Get number of GPUs
print(f"Number of GPUs: {num_gpus}")



import os
env = os.environ.copy()  
env["CUDA_VISIBLE_DEVICES"] = "0"  
print(env["CUDA_VISIBLE_DEVICES"])
