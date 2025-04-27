import torch
from extended_stp import *
from plotting import *
from data_setup import *

# Recreates results from paper
def simulate_paper(input_length=5, N=5000, P=16, f=0.05, dt=1e-4):
    # Initialize model with paper parameters
    model = PaperSTPWrapper(
        N=N, P=P, f=f, dt=dt,
        J_EE=8.0, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, I_b = 8.0
    ).to(device)
    input_strength = 365.0 # Pt. 2.3 of supplemental material
    duration = 2.5
    simulate_paper_with_model(model, input_strength, duration, input_length)

def simulate_paper_extended(input_length=5, N_a=5000, N_b=5000, P=16, f=0.05,
                            dt=1e-4, duration=2.5, input_strength=365.0, **kwargs):
    # Initialize model with paper parameters
    model = ExtendedSTPWrapper(
        N_a=N_a, N_b=N_b, P=P, f=f, dt=dt,
        J_EE=8.0, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, I_b = 8.0, **kwargs
    ).to(device)
    simulate_paper_with_model(model, input_strength, duration, input_length)

def simulate_cluster_stp(dt=1e-3):
    model = ClusterSTPWrapper(P=16, f=0.05, dt=dt, J_EE=8.0, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, J_EI=1.1, I_b = 8.0).to(device)
    simulate_paper_with_model(model, input_strength=225.0, duration=2.5)

def simulate_paper_with_model(model:STPWrapper, input_strength, duration=2.5, input_length=5):
    """Simulate the paper test with the given model and input strength, for 'duration' seconds."""
    # Stimulation sequence
    input_idx = torch.Tensor([[*range(input_length)]])
    inputs = generate_one_hot_impulses(input_idx, model.P, model.dt, impulse_strength=input_strength)
    inputs = pad_impulses(inputs, model.dt, duration)
    seq_len = int(duration / model.dt)

    # Run simulation
    states, outputs = model(inputs)
    # (state index, seq_length, batch, neuron)

    # TODO Fixing cluster plotting
    state_traces = {}
    state_vars = ('h', 'u', 'x', 'h_I')
    per_neuron_vars = ('h', 'u', 'x')

    cluster_neurons = torch.zeros(input_length, model.N, dtype=torch.bool)
    for p in range(input_length):
        cluster_neurons[p, :len(model.eta[p])] = model.eta[p].bool()

    for i, variable in enumerate(per_neuron_vars): # Variables with per-neuron values
        var_vals = states[i][:, 0, :] # [seq_len x neuron]
        state_traces[variable] = torch.zeros(input_length, seq_len)
        for p in range(input_length):
            state_traces[variable][p, :] = torch.mean(var_vals[:, cluster_neurons[p]], dim=1)
        state_traces[variable] = state_traces[variable].tolist()

    state_traces['h_I'] = states[3][:, 0].tolist()

    ux_traces = torch.zeros(input_length, seq_len)
    u, x = states[1], states[2]
    u_x = u[:, 0, :] * x[:, 0, :]
    for p in range(input_length):
        ux_traces[p, :] = torch.mean(u_x[:, cluster_neurons[p]], dim=1)
    ux_traces = ux_traces.tolist()


    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    # Plot state variables in a 2x2 figure
    titles = ['Mean h (Cluster)', 'Mean u (Cluster)', 'Mean x (Cluster)', 'h_I']

    for i, variable in enumerate(state_vars):
        ax = axes[i // 2, i % 2]
        if variable in per_neuron_vars:
            for p in range(input_length):
                ax.plot(state_traces[variable][p], label=f'Cluster {p+1}')
        else:  # Scalar variable
            ax.plot(state_traces[variable], label=variable)
        ax.set_title(titles[i])
        ax.set_xlabel('Time (0.1ms steps)')
        ax.set_ylabel(variable)
        ax.legend()

    # Plot ux
    ax = axes[2, 0]
    for p in range(input_length):
        ax.plot(ux_traces[p], label=f'Cluster {p+1}')
    ax.set_xlabel(f'Time ({model.dt * 1e3}ms steps)')
    ax.set_ylabel('Synaptic Efficacy (u*x)')
    ax.legend()

    # Hide unused plot
    axes[2, 1].set_visible(False)

    plt.tight_layout()
    plt.show()

# ---- Tests ----


# ---- Main ----
if __name__ == "__main__":
    # simulate_cluster_stp()
    # simulate_paper_extended(input_length=4, N_a=1000, N_b=1000, dt=1e-3, input_strength=225.0)
    # 001_parity model 04/27: Clusters firing in sync, not correct
    # simulate_paper_extended(P=2, f=0.4, input_length=2, N_a=100, N_b=200, dt=1e-3, duration=1.0)
    # Testing better values: correct firing
    simulate_paper_extended(P=16, f=0.05, input_length=2, N_a=1000, N_b=200, dt=1e-3, duration=1.0)
    # train_xor()