# From https://wandb.ai/ayush-thakur/dl-question-bank/reports/How-To-Check-If-PyTorch-Is-Using-The-GPU--VmlldzoyMDQ0NTU

import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# get index of currently selected device
print(torch.cuda.current_device())

# get number of GPUs available
print(torch.cuda.device_count())

# get the name of the device
print(torch.cuda.get_device_name(0))