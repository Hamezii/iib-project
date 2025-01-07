"""Simple and min GRU models trained on a sequence memory task"""

import math

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

# ---- Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEBUG below:
# device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)


# ---- Constants
NUM_CLASSES = 10
INPUT_SIZE = NUM_CLASSES + 1
HIDDEN_SIZE = 32

LEARNING_STEPS = 4000
LEARNING_RATE = 0.05


# ---- Task data
BATCH_SIZE = 32
SEQ_LENGTH = 8
INPUT_STEP_T = 1 # tensor.repeat_interleave(N) repeats values in a tensor
T_AFTER_GO_DELAY = 0
T_SEQUENCE = SEQ_LENGTH * INPUT_STEP_T
T_FULL_INPUT = (SEQ_LENGTH+1) * INPUT_STEP_T
T_OUTPUT_START = T_SEQUENCE + T_AFTER_GO_DELAY

NOISE_STD = 0.05

class RandomSequenceGenerator:
    """Generates random input sequences"""

    def __next__(self):
        # Random integers from zero to NUM_CLASSES - 1, length SEQ_LENGTH
        seq_indices = torch.randint(0, NUM_CLASSES, (SEQ_LENGTH,)).to(device)
        # Append a final number to the end of the sequence, the "go signal"
        # comma is here to stop it being 0-dimensional and allow it to concat
        go_signal = torch.tensor((NUM_CLASSES, )).to(device)
        x_indices = torch.cat((seq_indices, go_signal)).to(device)

        # Getting one-hot sequence from index tensor
        # pylint: disable=E1102
        x = F.one_hot(x_indices, num_classes=NUM_CLASSES+1) \
            .float() \
            .repeat_interleave(INPUT_STEP_T, dim=0) \
            .to(device)

        assert seq_indices.shape == (SEQ_LENGTH,)
        assert x.shape == (T_FULL_INPUT, NUM_CLASSES+1), x.shape
        return x, seq_indices

# pylint: disable=W0223


class SequenceDataset(IterableDataset):
    """This is passed to DataLoader to create minibatches"""

    def __iter__(self):
        return RandomSequenceGenerator()


# ---- Models
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset gate
        self.W_r = torch.nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.P_r = torch.nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))

        # Update gate z_t
        # W_z and P_z are set to 0 and not trained
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))

        # Internal state
        # r_t is passed in forward method

        # Randomise parameters
        scaled_mag = 1/math.sqrt(hidden_size)
        for _, param in self.named_parameters():
            nn.init.uniform_(param, a=-(scaled_mag), b=(scaled_mag))

        # Activation/transfer functions
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, x, r_t_prev):
        self.z_t = self.Sigmoid(self.b_z)
        # print("x.shape:", x.shape)
        # print("r_t_prev.shape:", r_t_prev.shape)
        # print("self.z_t.shape:", self.z_t.shape)
        # print("self.W_r.shape:", self.W_r.shape)
        # print("self.P_r.shape:", self.P_r.shape)
        r_t = (1 - self.z_t) * r_t_prev + self.z_t * \
            self.Tanh((self.W_r @ r_t_prev.T).T +
                      (self.P_r @ x.T).T + self.b_r)

        return r_t


class MinGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # h_hat_t gate
        self.lin_h_hat = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.lin_h_hat.weight)
        nn.init.zeros_(self.lin_h_hat.bias)

        # z_t update gate
        self.lin_z = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.lin_z.weight)
        nn.init.zeros_(self.lin_z.bias)

        # Activation/transfer functions
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, h_t_prev):
        # assert x.shape == (BATCH_SIZE, INPUT_SIZE)
        # assert h_t_prev.shape == (BATCH_SIZE, HIDDEN_SIZE)
        z_t = self.Sigmoid(self.lin_z(x))
        h_hat_t = self.lin_h_hat(x)
        h_t = (1 - z_t) * h_t_prev + z_t * h_hat_t
        return h_t


class MemoryModel(nn.Module):
    def __init__(self, cell=MinGRUCell):
        super().__init__()
        # GRU cell to run iteratively through time
        self.grucell = cell(INPUT_SIZE, HIDDEN_SIZE)
        # Mapping final internal cell values to class decisions
        # self.C = torch.nn.Parameter(torch.zeros(NUM_CLASSES, HIDDEN_SIZE))
        # nn.init.xavier_uniform_(self.C)
        self.out_layer = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

        # Pooling ys for each input element
        self.pool = nn.AvgPool1d(kernel_size=INPUT_STEP_T)

    def forward(self, x):
        """Runs the GRU cell once for each step."""
        # assert x.shape == (BATCH_SIZE, SEQ_LENGTH+1, NUM_CLASSES+1)

        # Initialise hidden state
        h_t = torch.zeros(x.shape[0], HIDDEN_SIZE).to(device)

        # Initial y matrix
        y = torch.zeros((x.shape[0], T_SEQUENCE, NUM_CLASSES)).to(device)

        # Zero matrix for no input
        zero_input = torch.zeros((x.shape[0], INPUT_SIZE)).to(device)

        # Iteratively modelling
        for i in range(T_OUTPUT_START + T_SEQUENCE):
            # Could experiment with replacing zero_input with prev:
            # prev = torch.cat((y[:, i - SEQ_LENGTH - 1, :], torch.zeros((x.shape[0], 1)).to(device)), dim=1).to(device)
            x_slice = x[:, i, :] if i < x.shape[1] else zero_input
            # Noise
            noise = NOISE_STD * torch.randn(x_slice.shape).to(device)
            h_t = self.grucell(x_slice+noise, h_t)
            if i >= T_OUTPUT_START:
                y[:, i - T_OUTPUT_START, :] = self.out_layer(h_t)

        y = y.transpose(1, 2)
        y = self.pool(y) # Averaging across individual inputs
        return y


dataset = SequenceDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE)
model = MemoryModel(GRUCell).to(device)
loss_func = nn.CrossEntropyLoss(ignore_index=INPUT_SIZE-1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.25, patience=10)

seq_length_min = 2
seq_length_max = 5
SEQ_LENGTH = seq_length_min
T_SEQUENCE = SEQ_LENGTH * INPUT_STEP_T
T_FULL_INPUT = (SEQ_LENGTH+1) * INPUT_STEP_T
T_OUTPUT_START = T_SEQUENCE + T_AFTER_GO_DELAY

LR_UPDATES = 50
i = 0
avg_loss = 0
learn_loss = 0


model.train()
for (x, seq_indices) in loader:

    assert x.shape == (BATCH_SIZE, T_FULL_INPUT, NUM_CLASSES+1)
    assert seq_indices.shape == (BATCH_SIZE, SEQ_LENGTH)

    model.train()
    outputs = model(x)
    assert outputs.shape == (BATCH_SIZE, NUM_CLASSES, SEQ_LENGTH)
    loss = loss_func(outputs, seq_indices)

    learn_loss += loss
    # Iterate through sequence lengths
    if SEQ_LENGTH == seq_length_max:
        SEQ_LENGTH = seq_length_min
        learn_loss /= 1 + (seq_length_max - seq_length_min)
        optimizer.zero_grad()
        learn_loss.backward()
        optimizer.step()
        learn_loss = 0
    else:
        SEQ_LENGTH += 1
    T_SEQUENCE = SEQ_LENGTH * INPUT_STEP_T
    T_FULL_INPUT = (SEQ_LENGTH+1) * INPUT_STEP_T
    T_OUTPUT_START = T_SEQUENCE + T_AFTER_GO_DELAY

    avg_loss += loss.item()
    if (i+1) % LR_UPDATES == 0:
        avg_loss /= LR_UPDATES
        print('Step [{}/{}], Loss: {:.4f}'
              .format(i + 1, LEARNING_STEPS, avg_loss))
        scheduler.step(avg_loss)
        if avg_loss < 0.01:
            break
        avg_loss = 0
    i += 1
    if i == LEARNING_STEPS:
        break




# "Validation"
# model.eval()
# with torch.no_grad():
#     for i in range(10):
#         (x, seq_indices) = next(iter(dataset))
#         x = x.unsqueeze(0) # i.e. 1 batch size
#         print(f"Input sequence: {seq_indices}")
#         outputs = model(x)
#         #outputs = nn.LogSoftmax(dim=1)(model(x))
#         print(f"Output sequence: {outputs}")

# Testing model
# model.eval()
# with torch.no_grad():
#     tot = 0
#     correct = 0
#     n_tests = 4000
#     print_step = 100
#     for i in range(n_tests):
#         SEQ_LENGTH = torch.randint(seq_length_min, seq_length_max + 1, (1,)).item()
#         T_SEQUENCE = SEQ_LENGTH * INPUT_STEP_T
#         T_FULL_INPUT = (SEQ_LENGTH+1) * INPUT_STEP_T
#         T_OUTPUT_START = T_SEQUENCE + T_AFTER_GO_DELAY

#         (x, seq_indices) = next(iter(dataset))
#         x = x.unsqueeze(0)  # i.e. 1 batch size
#         outputs = model(x)
#         _, predicted = torch.max(outputs.data, 1)

#         tot += SEQ_LENGTH
#         correct = correct + (predicted == seq_indices).sum().item()
#         if (i+1) % print_step == 0:
#             print(f"Test {i+1}:\n  Input sequence: \t{seq_indices.tolist()}")
#             print(f"  Output sequence: \t{predicted.squeeze().tolist()}")

#     accuracy = 100 * correct / tot
#     print(f"Test accuracy of the model on {n_tests} sequences: {accuracy} %")


# Testing per length:
model.eval()
with torch.no_grad():
    A = []
    for seq_len in range(seq_length_min-1, seq_length_max + 3):
        tot = 0
        correct = 0
        n_tests = 1000
        print_step = 500
        for i in range(n_tests):
            SEQ_LENGTH = seq_len
            T_SEQUENCE = SEQ_LENGTH * INPUT_STEP_T
            T_FULL_INPUT = (SEQ_LENGTH+1) * INPUT_STEP_T
            T_OUTPUT_START = T_SEQUENCE + T_AFTER_GO_DELAY

            (x, seq_indices) = next(iter(dataset))
            x = x.unsqueeze(0)  # i.e. 1 batch size
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)

            tot += SEQ_LENGTH
            correct = correct + (predicted == seq_indices).sum().item()
            if (i+1) % print_step == 0:
                print(f"Test {i+1}:\n  Input sequence: \t{seq_indices.tolist()}")
                print(f"  Output sequence: \t{predicted.squeeze().tolist()}")

        accuracy = 100 * correct / tot
        A.append(accuracy)
        print(f"Test accuracy of the model on {n_tests} sequences, length {seq_len}: {accuracy} %")
    print(A)
