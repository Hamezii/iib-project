# Methods for generating data
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

class SequenceMemoryDataGenerator:
    """Iterable that generates random sequence memory training data"""
    def __init__(self, batch_size, seq_length, channels):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.channls = channels

    def __next__(self):
        """Returns sequence, test item, and the item following it in the sequence.
        
        Each element in the sequence is a unique number from 0 to channels-1
        The test item is a random element from the sequence (not last).
        The target is the next item in the sequence after the test item.
        """
        # 
        seq = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long).to(device)
        test = torch.zeros(self.batch_size, dtype=torch.long).to(device)
        target = torch.zeros(self.batch_size, dtype=torch.long).to(device)

        for b in range(self.batch_size):
            shuffled = torch.randperm(self.channls)
            seq[b, :] = shuffled[:self.seq_length]
            test_id = torch.randint(0, self.seq_length-1, (1,)).item()
            test[b] = seq[b, test_id]
            target[b] = seq[b, test_id + 1]

        return seq, test, target
