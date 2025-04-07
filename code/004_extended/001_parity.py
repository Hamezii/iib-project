
import torch
from torch import nn, optim
from extended_stp import *
from plotting import *
from data_setup import *
import train_parity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8
P = 2

# Times
DT = 1e-3
DURATION = 1.5 #s
AVG_OVER_LAST = 0.2 #s

PARITY_IMPULSES = 2

LEARNING_STEPS = 200
LEARNING_RATE = 1e-3
# TODO Try making B weights learnable
model = ExtendedSTPWrapper(N_a=100, N_b=100, P=P, f=0.4, out_size=2, dt=DT).to(device)

data_iter = train_parity.ParityDataGenerator(BATCH_SIZE, PARITY_IMPULSES)
parity_dataloader = get_dataloader_from_iterable(data_iter)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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

        avg_idx = int(AVG_OVER_LAST / DT)
        outputs_averaged = torch.mean(outputs[-avg_idx:, :, :], dim=0)

        # TEMP fixing target
        # target = torch.ones_like(target)

        loss = loss_func(outputs_averaged, target)
        print(f"Step {i}, loss = {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(states)
        # plot_impulses(states[0][:, :, :4], DT, 0) # Plot 4 neurons of h
        # plot_impulses(outputs, DT, 0)

        if i+1 == LEARNING_STEPS or loss < 0.01:
            for b in range(BATCH_SIZE):
                print(f"Batch {b} input: {inp_seq[b]}")
                plot_impulses(outputs, DT, b)
            break
except KeyboardInterrupt:
    print(f"Batch 0 input: {inp_seq[0]}")
    plot_impulses(outputs, DT, 0)