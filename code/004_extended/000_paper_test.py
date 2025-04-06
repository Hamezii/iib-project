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

def simulate_paper_extended(input_length=5, N_a=5000, N_b=5000, P=16, f=0.05, dt=1e-4):
    # Initialize model with paper parameters
    model = ExtendedSTPWrapper(
        N_a=N_a, N_b=N_b, P=P, f=f, dt=dt,
        J_EE=8.0, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, I_b = 8.0
    ).to(device)
    input_strength = 365.0 # Pt. 2.3 of supplemental material
    duration = 2.5
    simulate_paper_with_model(model, input_strength, duration, input_length)

def simulate_cluster_stp():
    model = ClusterSTPWrapper(P=16, f=0.05, dt=1e-4, J_EE=8.0, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, J_EI=1.1, I_b = 8.0).to(device)
    simulate_paper_with_model(model, input_strength=225.0, duration=2.5)

def simulate_paper_with_model(model:STPWrapper, input_strength, duration=2.5, input_length=5):
    """Simulate the paper test with the given model and input strength, for 'duration' seconds."""
    # Stimulation sequence
    input_idx = torch.Tensor([[*range(5)]])
    inputs = generate_one_hot_impulses(input_idx, 16, model.dt)
    inputs = pad_impulses(inputs, model.dt, 2.5)
    seq_len = int(duration / model.dt)

    # Run simulation
    states, outputs = model(inputs)
    # (seq_len, state index, batch, neuron)
    ux_traces = {p: [] for p in range(input_length)}
    state_vars = ('h', 'u', 'x', 'h_I')
    scalar_vars = ('h_I',)
    per_neuron_vars = ('h', 'u', 'x')
    state_traces = {}
    for variable in per_neuron_vars: # Variables with per-neuron values
        state_traces[variable] = {p: [] for p in range(input_length)}
    for variable in scalar_vars: # Variables with scalar values
        state_traces[variable] = []

    for t in range(seq_len):
        state = states[t]
        for i, var in enumerate(state_vars):
            if var in per_neuron_vars: # If variable has per-neuron values
                var_vals = state[i][0, :]
                for p in range(input_length):
                    cluster_neurons = torch.zeros(model.N, dtype=torch.bool)
                    cluster_neurons[:len(model.eta[p])] = model.eta[p].bool()
                    # cluster_neurons = model.eta[p].bool()
                    state_traces[var][p].append(var_vals[cluster_neurons].mean().item())
            else: # If variable is scalar
                state_traces[var].append(state[i].item())

        u, x = state[1], state[2]
        u_x = u[0, :] * x[0, :]
        assert u_x.shape == (model.N,), f"u_x shape: {u_x.shape}"
        for p in range(input_length):
            cluster_neurons = torch.zeros(model.N, dtype=torch.bool)
            cluster_neurons[:len(model.eta[p])] = model.eta[p].bool()
            # cluster_neurons = model.eta[p].bool()
            ux_traces[p].append(u_x[cluster_neurons].mean().item())

    # Plot results
    plt.figure()
    for p in range(input_length):
        plt.plot(ux_traces[p], label=f'Cluster {p+1}')
    plt.xlabel(f'Time ({model.dt * 1e3}ms steps)')
    plt.ylabel('Synaptic Efficacy (u*x)')
    plt.legend()
    plt.show()

    # Plot state variables in a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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

    plt.tight_layout()
    plt.show()

# ---- Tests ----


# ---- Main ----
if __name__ == "__main__":
    # simulate_cluster_stp()
    simulate_paper_extended(input_length=4, N_a=1000, N_b=1, dt=1e-3)
    # train_xor()