"""Simple GRU implementation on MNIST dataset"""

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
LEARNING_RATE = 0.01
NUM_EPOCHS = 6


# ---- Task data
SEQ_LENGTH = 8
BATCH_SIZE = 64

class RandomSequenceGenerator:
    """Generates random input sequences"""
    def __next__(self):
        # Random integers from zero to NUM_CLASSES - 1, length SEQ_LENGTH
        seq_indices = torch.randint(0, NUM_CLASSES, (SEQ_LENGTH,)).to(device)
        # Append a final number to the end of the sequence, the "go signal"
        go_signal = torch.tensor((NUM_CLASSES, )).to(device) # comma is here to stop it being 0-dimensional and allow it to concat
        x_indices = torch.cat((seq_indices, go_signal)).to(device)

        # Getting one-hot sequence from index tensor
        #pylint: disable=E1102
        x = F.one_hot(x_indices, num_classes=NUM_CLASSES+1).float().to(device)

        assert seq_indices.shape == (SEQ_LENGTH,)
        assert x.shape == (SEQ_LENGTH + 1, NUM_CLASSES+1)
        return x, seq_indices

#pylint: disable=W0223
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
        r_t = (1 - self.z_t) * r_t_prev + self.z_t * self.Tanh((self.W_r @ r_t_prev.T).T + (self.P_r @ x.T).T + self.b_r)
        
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

    def forward(self, x):
        """Runs the GRU cell once for each step."""
        # assert x.shape == (BATCH_SIZE, SEQ_LENGTH+1, NUM_CLASSES+1)


        # Initialise hidden state
        h_t = torch.zeros(x.shape[0], HIDDEN_SIZE).to(device)

        # Initial y matrix
        y = torch.zeros((x.shape[0], SEQ_LENGTH, NUM_CLASSES)).to(device)

        # Zero matrix for no input
        zero_input = torch.zeros((x.shape[0], INPUT_SIZE)).to(device)

        # Iteratively modelling
        for i in range(SEQ_LENGTH * 2):
            # Could experiment with replacing zero_input with y[:, i - SEQ_LENGTH, :]
            x_slice = x[:, i, :] if i < x.shape[1] else zero_input
            h_t = self.grucell(x_slice, h_t)
            if i >= SEQ_LENGTH: # SEQ_LENGTH + 1 - 1
                y[:, i - SEQ_LENGTH, :] = self.out_layer(h_t)

        return y


dataset = SequenceDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE)
model = MemoryModel(GRUCell).to(device)
loss_func = nn.CrossEntropyLoss(ignore_index=INPUT_SIZE-1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

i = 0
step_total = 1500
for (x, seq_indices) in loader:
    assert x.shape == (BATCH_SIZE, SEQ_LENGTH + 1, NUM_CLASSES+1)
    assert seq_indices.shape == (BATCH_SIZE, SEQ_LENGTH)

    model.train()
    outputs = model(x)
    loss = loss_func(outputs.transpose(1, 2), seq_indices)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
        print ('Step [{}/{}], Loss: {:.4f}' 
                .format(i + 1, step_total, loss.item()))

    i += 1
    if i == step_total:
        break

# Testing
model.eval()
with torch.no_grad():
    for i in range(10):
        (x, seq_indices) = next(iter(dataset))
        x = x.unsqueeze(0)
        print(f"Input sequence: {seq_indices}")
        outputs = model(x)
        print(f"Output sequence: {outputs}")



# For learning, use CrossEntropyLoss with ignore_index (look at nn.Loss autocomplete for other methods)
# nn.CrossEntropyLoss(ignore_index=NUM_CLASSES)


# # ---- Training
# def train(num_epochs, model, loss_func, optimizer, train_loader, test_loader):
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

#     step_total = len(train_loader)

#     for epoch in range(num_epochs):
#         # Training
#         model.train()
#         for i, (images, labels) in enumerate(train_loader):

#             images = images.squeeze(1).to(device)
#             # print("images.shape:", images.shape)
#             labels = labels.to(device)

#             # Forward pass
#             outputs = model(images)
#             loss = loss_func(outputs, labels)

#             # BPTT and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch + 1, num_epochs, i + 1, step_total, loss.item()))

#         # Validation step
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images = images.squeeze(1).to(device)
#                 labels = labels.to(device)
#                 outputs = model(images)
#                 val_loss += loss_func(outputs, labels).item()

#         val_loss /= len(test_loader)
#         print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss}')

#         # Step scheduler (modifying learning rate) based on validation loss
#         scheduler.step(val_loss)



# # ---- Testing model
# if __name__ == "__main__":
#     model = SequentialGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, MinGRUCell).to(device)
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     train(NUM_EPOCHS, model, loss_func, optimizer, train_loader, test_loader)

#     # Testing model
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.squeeze(1).to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total = total + labels.size(0)
#             correct = correct + (predicted == labels).sum().item()
#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
