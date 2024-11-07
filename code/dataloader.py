"""Helper module for formatting data
Currently redundant.
"""


from torch.utils.data import DataLoader

def get_data_loader(data,
    batch_size:int,
    shuffle=True):
    
    return DataLoader(data, batch_size, shuffle)