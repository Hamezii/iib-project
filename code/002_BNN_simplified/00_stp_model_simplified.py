# This synaptic model uses the simplified STP model equations 12 to 15
# Equation Source: Supplemental Information - Synaptic Correlates of Working Memory Capacity

import numpy as np
import matplotlib.pyplot as plt

class ClusterSTPModel:
    def __init__(self, P=10, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_EE=7.5, f=0.05, 
                 J_EI=1.1, J_IE=2.2, alpha=1.5, I_b=8):
        """
        Initialize the simplified STP model (equations 12 to 15)
        for P memory clusters.
        
        Parameters:
          alpha (float): Gain parameter for firing rate function.
        Single-cell parameters:
          tau (float): Membrane time constant (s).
        Network parameters:
          f (float): Sparseness parameter (scales cross-cluster connections).
          P (int): Number of memory clusters.
        Synaptic parameters:
          J_IE (float): Synaptic efficacy E -> I (mV).
          J_EI (float): Synaptic efficacy I -> E (mV).
          J_EE (float): Excitatory synaptic strength within a cluster.
          I_b (float): Background input current.
        Short-term synaptic dynamic parameters:
          U (float): Baseline utilization factor.
          tau_f (float): Recovery time of utilisation factor (s).
          tau_d (float): Recovery time of synaptic resources (s).

        """
        self.P = P
        self.U = U
        self.tau = tau
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.J_EE = J_EE
        self.f = f
        self.J_EI = J_EI
        self.J_IE = J_IE
        self.alpha = alpha
        self.I_b = I_b

        # Initialize variables
        self.h = np.zeros(P)        # Synaptic currents (h_rho)
        self.u = np.full(P, U)      # Facilitation variables (u_rho)
        self.x = np.ones(P)         # Depression variables (x_rho)
        self.h_I = 0.0              # Inhibitory population current

        # Effective connectivity matrix J (P x P)
        # Diagonal = J_EE, off-diagonal = f*J_EE
        self.J = np.full((P, P), f * J_EE)
        np.fill_diagonal(self.J, J_EE) 

    def compute_R(self, h):
        """Compute firing rate R(h) = alpha * ln(1 + exp(h / alpha))"""
        return self.alpha * np.log(1 + np.exp(h / self.alpha))

    def update(self, dt, I_e=0.0):
        """
        Update population variables for one time step.
        
        Parameters:
        dt (float): Integration time step (s).
        I_e (float or array): External input current (scalar or per-cluster).
        
        Returns:
        u_x (np.ndarray): Synaptic efficacies (u_rho * x_rho) for each cluster.
        """
        # Firing rates for excitatory clusters and inhibitory population
        R = self.compute_R(self.h)
        R_I = self.compute_R(self.h_I)        

        # Update excitatory cluster variables
        # Equation (12)
        synaptic_inputs = self.J @ (self.u * self.x * R)
        dh = -self.h + synaptic_inputs - self.J_EI * R_I + self.I_b + I_e
        dh /= self.tau
        # Equation (13)
        du = (self.U - self.u) / self.tau_f + self.U * (1 - self.u) * R
        # Equation (15)
        dx = (1 - self.x) / self.tau_d - self.u * self.x * R

        # Update inhibitory population (Equation 15)
        dh_I = -self.h_I + self.J_IE * np.sum(R)
        dh_I /= self.tau

        # Euler integration step
        self.h += dh * dt
        self.u += du * dt
        self.x += dx * dt
        self.h_I += dh_I * dt

        return self.u * self.x  # Return synaptic efficacies

def test_model():
    model = ClusterSTPModel()
    dt = 1e-3
    t_dur = 10 # Duration of loading external stimulus (ms)
    u_x = []
    for i in range(2000):
        # Testing small input to first cluster
        I_e = np.zeros(model.P)
        if i< t_dur:
            I_e[0] = 565.0
        u_x.append(model.update(dt, I_e=I_e))

    print(f"Synaptic efficacies:")
    print(f"t = 0ms: {u_x[0]}")
    print(f"t = 10ms: {u_x[t_dur - 1]}")
    print(f"t = 500ms: {u_x[499]}")
    print(f"t = 1000ms: {u_x[999]}")
    print(f"t = end: {u_x[-1]}")


    plt.plot(np.arange(0, 2000, 1), [eff[0] for eff in u_x], label="Triggered population")
    plt.plot(np.arange(0, 2000, 1), [eff[1] for eff in u_x], label="Control population")
    #plt.ylim(45, 100)
    #plt.xlim(0, 80)
    plt.xlabel("Time step")
    plt.ylabel("Efficacy")
    #plt.title()
    plt.legend()
    #plt.grid(True)
    #plt.savefig("000_efficacy_test.svg")
    plt.show()

def test_model_firing_rate():
    inputs = 5
    model = ClusterSTPModel(J_EE=8, tau_f=1.5, tau_d=0.3, U=0.3, tau=8e-3,
                            J_IE=1.75, J_EI=1.1, alpha=1.5, P=16, I_b=8)
    dt = 1e-4
    t_dur = 300 # Duration of loading external stimulus (dt steps)
    t_spacing = 1000
    R = []
    u_x = []

    for i in range(inputs):
        t = t_spacing
        for _ in range(t_spacing):
            t -= 1
            I_e = np.zeros(model.P)
            if t < t_dur:
                I_e[i] = 225.0
            u_x.append(model.update(dt, I_e=I_e))
            R.append(model.compute_R(model.h))
    for i in range(20000):
        I_e = np.zeros(model.P)
        u_x.append(model.update(dt, I_e=I_e))
        R.append(model.compute_R(model.h))

    plt.figure(figsize=(25, 5))
    for i in range(inputs):
        plt.plot([r[i] for r in R], label=f"Cluster {i}")
    #plt.ylim(45, 100)
    #plt.xlim(0, 80)
    plt.xlabel("Time step")
    plt.ylabel("Firing rate R")
    #plt.title()
    plt.legend()
    #plt.grid(True)
    #plt.savefig("001_test_R.svg")
    #plt.savefig("001_test_R.png")
    plt.show()

    plt.figure(figsize=(25, 5))
    for i in range(inputs):
        plt.plot([r[i] for r in u_x], label=f"Cluster {i}")
    #plt.ylim(45, 100)
    #plt.xlim(0, 80)
    plt.xlabel("Time step")
    plt.ylabel("Efficacy ux")
    #plt.title()
    plt.legend()
    #plt.grid(True)
    #plt.savefig("002_test_ux.svg")
    #plt.savefig("002_test_ux.png")
    plt.show()

if __name__ == "__main__":
    test_model_firing_rate()