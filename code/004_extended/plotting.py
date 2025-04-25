# Helper functions for plotting things
import torch
import matplotlib.pyplot as plt

VERBOSE = False

# def plot_memory_neuron_state(model, states, save=None):
#     # Plot state variables in a 2x2 figure
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     titles = ['Mean h (Cluster)', 'Mean u (Cluster)', 'Mean x (Cluster)', 'h_I']

#     for i, variable in enumerate(state_vars):
#         ax = axes[i // 2, i % 2]
#         if variable in per_neuron_vars:
#             for p in range(input_length):
#                 ax.plot(state_traces[variable][p], label=f'Cluster {p+1}')
#         else:  # Scalar variable
#             ax.plot(state_traces[variable], label=variable)
#         ax.set_title(titles[i])
#         ax.set_xlabel('Time (0.1ms steps)')
#         ax.set_ylabel(variable)
#         ax.grid(alpha=0.15)
#         ax.legend()

#     plt.tight_layout()
#     if save is not None:
#         plt.savefig(save)
#         if VERBOSE:
#             print(f"Saving to {save}")
#     if show:
#         plt.show()
#     plt.close()


def plot_impulses(impulses, dt, batch_id, y_label="Input", title=None, show=True, save=None):
    """Plot impulses (time_i, batch_size, channels).
    
    Assumes impulses don't include last time step.
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
    plt.title(title)

    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        if VERBOSE:
            print(f"Saving to {save}")
    if show:
        plt.show()
    plt.close()
