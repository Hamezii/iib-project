# Helper functions for plotting things
import torch
import matplotlib.pyplot as plt

# TODO add dt
def plot_impulses(impulses, dt, batch_id):
    """Plot impulses (time_i, batch_size, channels).
    
    Assumes impulses don't include 0 time step.
    """
    input_len, batch_size, channels = impulses.shape
    # NOTE Need to be careful with times indexes.
    times_ms = torch.linspace(dt * 1e3, input_len * dt * 1e3, input_len)
    plt.figure()
    for p in range(channels):
        plt.plot(times_ms, impulses[:, batch_id, p].tolist(), label=f'Channel {p}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input')
    plt.legend()
    plt.show()