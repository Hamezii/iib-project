# TODO Plot loss vs time step
# TODO Get readout of tsodyks network, to check its working correctly

import os
import torch
from torch import nn, optim
from extended_stp import *
from plotting import *
from data_setup import *
import data_gen

OUT_DIR = "OUT/"
SAVE_DIR = input("Save to OUT/<name>/ (leave empty for no saving): ")
if SAVE_DIR:
    SAVE_DIR = OUT_DIR + SAVE_DIR + "/"
    directory = os.path.dirname(SAVE_DIR)
    os.makedirs(directory, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model values
P = 16
f = 0.05
I_b = 1.5

# Input data
IMPULSE_STRENGTH = 365.0
BATCH_SIZE = 16
INPUT_LENGTH = 3
ALPHABET_SIZE = 5 #P

# Times
DT = 1e-3
IMPULSE_DURATION = 30e-3 #s
IMPULSE_SPACING = 100e-3 #s
TIME_GAP_BEFORE_TEST = 0.2 #s
TIME_GAP_AFTER_TEST = 0.2 #s
AVG_OVER_LAST = 0.2 #s

DURATION_TO_TEST = (INPUT_LENGTH-1)*IMPULSE_SPACING + IMPULSE_DURATION + TIME_GAP_BEFORE_TEST
DURATION = DURATION_TO_TEST + IMPULSE_DURATION + TIME_GAP_AFTER_TEST + AVG_OVER_LAST

# Learning
LOAD_MODEL = False # "OUT/102/model.pth"
LEARNING = True
LEARNING_STEPS = 5000
LEARNING_RATE = 1e-3
EPOCH_STEPS = 5
MIN_LOSS = 0.1 #0.01

model = ExtendedSTPWrapper(N_a=1000, N_b=1000, P=P, f=f, out_size=P, dt=DT, I_b=I_b).to(device)
if LOAD_MODEL:
    model.load_state_dict(torch.load(LOAD_MODEL))
if not LEARNING:
    model.eval()
data_iter = data_gen.SequenceMemoryDataGenerator(BATCH_SIZE, INPUT_LENGTH, ALPHABET_SIZE)
dataloader = get_dataloader_from_iterable(data_iter)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
accumulated_loss = 0

# Cached data for plotting
CACHE_inp_seq_id = None
CACHE_test = None
CACHE_target = None
CACHE_inp = None
losses = []

def plot_responses():
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.legend()
    plt.grid(alpha=0.15)
    plt.ylim(bottom=0)
    plt.tight_layout()
    if SAVE_DIR:
        plt.savefig(SAVE_DIR + "losses.png")
    plt.show()

    for b in range(BATCH_SIZE):
        inp_string = ", ".join(str(int(a)) for a in CACHE_inp_seq_id[b])
        tst = CACHE_test[b]
        tgt = CACHE_target[b]
        print(f"Batch {b} input: {CACHE_inp_seq_id[b]}, test: {tst}, target: {tgt}")
        plot_impulses(outputs, DT, b,
                      y_label="Output",
                      title=f"Input: {inp_string}, Test: {tst}",
                      save= None if (not SAVE_DIR) else SAVE_DIR + inp_string+" t"+str(tst.item())+" OUT")
        plot_impulses(CACHE_inp[:, :, :ALPHABET_SIZE], DT, b,
                      y_label="Input",
                      title=f"Input: {inp_string}, Test: {tst}",
                      save= None if (not SAVE_DIR) else SAVE_DIR + inp_string+" t"+str(tst.item())+" IN")

    # TODO saving model, don't know if this is functional
    if SAVE_DIR:
        torch.save(model.state_dict(), SAVE_DIR + "model.pth")

try:
    for i, (inp_seq_id, test, target) in enumerate(dataloader):
        # inp_seq_id shape [batch x input_length]
        # test shape [batch]
        # target shape [batch]

        # [time_steps x batch x channel]
        inp_seq = generate_one_hot_impulses(
            inp_seq_id, P, DT,
            IMPULSE_STRENGTH, IMPULSE_SPACING, IMPULSE_DURATION
        )
        inp_test = generate_one_hot_impulses(
            test.unsqueeze(1), P, DT,
            IMPULSE_STRENGTH, IMPULSE_SPACING, IMPULSE_DURATION
        )
        inp = pad_impulses(inp_seq, DT, DURATION_TO_TEST)
        inp = torch.cat((inp, inp_test), dim=0)
        inp = pad_impulses(inp, DT, DURATION)

        # print(model.state_dict())
        # for p in model.parameters():
        #     print(p)
        #     print(p.device)
        states, outputs = model(inp)
        # outputs shape [time_steps x batch x channel]
        outputs = outputs[:, :, :ALPHABET_SIZE]
        # HACK so input plot is synced with output at force close
        CACHE_inp = inp
        CACHE_inp_seq_id = inp_seq_id
        CACHE_test = test
        CACHE_target = target

        avg_idx = int(AVG_OVER_LAST / DT)
        outputs_averaged = torch.mean(outputs[-avg_idx:, :, :], dim=0)

        # target = torch.ones_like(target) # TEMP fixing target

        # TODO could try error from target instead
        loss = loss_func(outputs_averaged, target)
        print(f"Step {i}, loss = {loss}")
        losses.append(loss.item())

        if i+1 == LEARNING_STEPS or loss < MIN_LOSS:
            plot_responses()
            break

        # TODO grad checking:
        # Could add time sequence of gradients to file,
        # or even absolute weight changes due to optimizer intricacies.
        if LEARNING:
            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                ...

            optimizer.step()

            accumulated_loss += loss.item()
            if i+1 % EPOCH_STEPS == 0:
                scheduler.step(accumulated_loss)
                accumulated_loss = 0

except KeyboardInterrupt:
    plot_responses()
