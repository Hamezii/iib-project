# TODO Plot loss vs time step
# TODO Get readout of tsodyks network, to check its working correctly

import os
import torch
from torch import nn, optim
from extended_stp import *
from plotting import *
from data_setup import *
import train_parity

OUT_DIR = "OUT/"
SAVE_DIR = input("Save to OUT/<name>/ (leave empty for no saving): ")
if SAVE_DIR:
    SAVE_DIR = OUT_DIR + SAVE_DIR + "/"
    directory = os.path.dirname(SAVE_DIR)
    os.makedirs(directory, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model values
P = 2
f = 0.4

# Times
DT = 1e-3
DURATION = 1.0 #s   # TODO Could try lowering this to see if it is able to work for shorter lengths
# Could also try with more elements and shorter time period
# DURATION = 0.13 + 0.2
AVG_OVER_LAST = 0.2 #s

# Input data
PARITY_IMPULSES = 3
BATCH_SIZE = 2 ** PARITY_IMPULSES
FIXED_DATA = True

# Learning
LEARNING_STEPS = 5000
LEARNING_RATE = 1e-3
EPOCH_STEPS = 2

model = ExtendedSTPWrapper(N_a=100, N_b=200, P=P, f=f, out_size=P, dt=DT).to(device)

data_iter = train_parity.ParityDataGenerator(BATCH_SIZE, PARITY_IMPULSES, FIXED_DATA)
parity_dataloader = get_dataloader_from_iterable(data_iter)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
accumulated_loss = 0

def plot_responses():
    for b in range(BATCH_SIZE):
        inp_string = "".join(str(int(a)) for a in inp_seq[b])
        print(f"Batch {b} input: {inp_seq[b]}")
        plot_impulses(outputs, DT, b, 
                      title=f"Input: {inp_string}",
                      save= None if (not SAVE_DIR) else SAVE_DIR + inp_string)

try:
    for i, (inp_seq, target) in enumerate(parity_dataloader):
        inp = generate_one_hot_impulses(inp_seq, P, DT).to(device)
        inp = pad_impulses(inp, DT, DURATION).to(device)
        # print(model.state_dict())
        # for p in model.parameters():
        #     print(p)
        #     print(p.device)
        states, outputs = model(inp)
        # outputs shape [time x batch x channel]
        outputs = outputs[:, :, :2]

        avg_idx = int(AVG_OVER_LAST / DT)
        outputs_averaged = torch.mean(outputs[-avg_idx:, :, :], dim=0)

        # TEMP fixing target
        # target = torch.ones_like(target)

        # TODO could try error from target instead
        loss = loss_func(outputs_averaged, target)
        print(f"Step {i}, loss = {loss}")

        # TODO grad checking:
        # Could add time sequence of gradients to file,
        # or even absolute weight changes due to optimizer intricacies.
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            ...

        optimizer.step()

        accumulated_loss += loss.item()
        if i+1 % EPOCH_STEPS == 0:
            scheduler.step(accumulated_loss)
            accumulated_loss = 0

        # print(states)
        # plot_impulses(states[0][:, :, :4], DT, 0) # Plot 4 neurons of h
        # plot_impulses(outputs, DT, 0)

        if i+1 == LEARNING_STEPS or loss < 0.01:
            plot_responses()
            break
except KeyboardInterrupt:
    plot_responses()
