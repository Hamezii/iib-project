# Methods for generating data for parity task
import torch
from torch.utils.data import DataLoader, IterableDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParityDataGenerator:
    """Iterable that generates random parity training data"""
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length

    def __next__(self):
        """Return sequences of ones and zeros, and a boolean of if there are an even number of ones."""
        seq = torch.randint(0, 2, (self.batch_size, self.seq_length)).to(device, dtype=torch.float32)
        parity = torch.sum(seq, dim=1) % 2 == 0
        assert seq.shape == (self.batch_size, self.seq_length), seq.shape
        assert parity.shape == (self.batch_size,), parity.shape
        return seq, parity
