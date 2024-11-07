"""Simple GRU implementation on MNIST dataset"""

import math

import matplotlib.pyplot as plt
import mnist_data
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


# ---- Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEBUG below:
# device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)


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

class SequentialGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # GRU cell to run iteratively through time
        self.grucell = GRUCell(input_size, hidden_size).to(device)
        # Mapping final internal cell values to class decisions 
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """Runs the GRU cell once for each slice of the image."""
        batch_size = x.size(0)
        r_t = torch.zeros(batch_size, self.grucell.hidden_size).to(device)

        for i in range(x.size(1)):
            x_slice = x[:,i,:]
            r_t = self.grucell(x_slice, r_t)
        out = self.linear(r_t)
        return out


# ---- Import data
BATCH_SIZE = 100

train_data = mnist_data.get_train_data()
test_data = mnist_data.get_test_data()

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=True)


# ---- Constants
SEQ_LENGTH = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 48
NUM_CLASSES = 10
LEARNING_RATE = 0.01
NUM_EPOCHS = 6


# ---- Training
def train(num_epochs, model, loss_func, optimizer, train_loader, test_loader):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    step_total = len(train_loader)

    for epoch in range(num_epochs):
        # Training
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            images = images.squeeze(1).to(device)
            # print("images.shape:", images.shape)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # BPTT and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, step_total, loss.item()))

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.squeeze(1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_loss += loss_func(outputs, labels).item()

        val_loss /= len(test_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss}')

        # Step scheduler (modifying learning rate) based on validation loss
        scheduler.step(val_loss)



# ---- Testing model
if __name__ == "__main__":
    model = SequentialGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(NUM_EPOCHS, model, loss_func, optimizer, train_loader, test_loader)

    # Testing model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
