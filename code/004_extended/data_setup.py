# Methods for managing data
import torch
from torch.utils.data import DataLoader, IterableDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_one_hot_impulses(
        idx_sequence, num_channels, dt,
        impulse_strength=365.0, impulse_spacing=100e-3, impulse_duration=30e-3):
    """
    Generate one hot impulses from a sequence of indices, size (time x batch_size x num_channels).
    
    idx_sequence: Matrix of indices (batch_size x length) to be converted to one-hot encoding.
    num_channels: total number of channels.
    dt: time step size.
    impulse_strength: Strength of input signal.
    impulse_spacing: Time between impulses (in seconds).
    impulse_duration: Duration of each impulse (in seconds).
    """
    batch_size, seq_len = idx_sequence.shape
    batch_indices = torch.arange(batch_size, device=device)
    i_spacing = int(impulse_spacing/dt)
    i_duration = int(impulse_duration/dt)
    data_len = i_spacing * seq_len
    impulses = torch.zeros(data_len, batch_size, num_channels, device=device)
    for i in range(seq_len):
        start = i * i_spacing
        end = start + i_duration
        impulses[start:end, batch_indices, idx_sequence[:,i].to(dtype=int)] = impulse_strength
    return impulses

def pad_impulses(impulses, dt, duration, pad_end=True):
    """Pad the impulses to the desired duration.
    
    impulses: Impulses of shape (time, batch_size, num_channels).
    dt: time step size.
    duration: Desired length of data (in seconds).
    pad_end: If True, pad at the end, else pad at the beginning.
    """
    data_len = int(duration / dt)
    padding = torch.zeros(data_len - impulses.shape[0],
                          impulses.shape[1], impulses.shape[2], device=device)
    if pad_end:
        return torch.cat((impulses, padding), dim=0)
    else:
        return torch.cat((padding, impulses), dim=0)

class CustomDataset(IterableDataset):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return self.iterable

def get_dataloader_from_iterable(iterable):
    """Get a DataLoader object from an iterable object, with automatic batching disabled."""
    dataset = CustomDataset(iterable)
    # batch_size = None disables automatic batching
    # However in this case, one can just use the data generator since loader is kind of useless now
    loader = DataLoader(dataset, batch_size=None)
    return loader