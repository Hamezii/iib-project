# Helper functions for plotting things
import torch
import matplotlib.pyplot as plt

# TODO add dt
def plot_impulses(impulses, dt, batch_id, y_label="Input", show=True):
    """Plot impulses (time_i, batch_size, channels).
    
    Assumes impulses don't include 0 time step.
    """
    input_len, batch_size, channels = impulses.shape
    # NOTE Need to be careful with times indexes.
    times_ms = torch.linspace(0, (input_len-1) * dt * 1e3, input_len)
    plt.figure()
    left, right = 0, round(input_len * dt * 1e3)
    plt.xlim(left, right)

    for p in range(channels):
        plt.plot(times_ms, impulses[:, batch_id, p].tolist(), label=f'Channel {p}')

    # Assert x ticks
    # plt.xticks(list(torch.linspace(left, right, steps=8)))
    #print(plt.xticks()[0])
    plt.xticks(list(plt.xticks()[0][1:-2]) + [left, right])

    plt.xlabel('Time (ms)')
    plt.ylabel(y_label)

    plt.grid(alpha=0.15)
    plt.legend()
    if show:
        plt.show()
