# Methods for generating data for parity task
import torch
from torch.utils.data import DataLoader, IterableDataset
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParityDataGenerator:
    """Iterable that generates random parity training data"""
    def __init__(self, batch_size, seq_length, fixed_data=False):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.fixed_data = fixed_data
        if self.fixed_data:
            assert self.batch_size == 2 ** self.seq_length, \
                "With fixed training data, batch_size must be consistent with sequence length."

    def __next__(self):
        """Return sequences of ones and zeros, and a boolean of if there are an even number of ones."""
        if self.fixed_data:
            seq = torch.zeros((2 ** self.seq_length, self.seq_length)).to(device, dtype=torch.float32)
            for i, vals in enumerate(product([0, 1], repeat=self.seq_length)):
                seq[i, :] = torch.Tensor(vals)
        else:
            seq = torch.randint(0, 2, (self.batch_size, self.seq_length)).to(device, dtype=torch.float32)
        parity = (torch.sum(seq, dim=1) % 2 == 0).to(device, torch.long)
        assert seq.shape == (self.batch_size, self.seq_length), seq.shape
        assert parity.shape == (self.batch_size,), parity.shape
        return seq, parity
