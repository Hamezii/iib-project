import numpy as np

class STPModel:
    def __init__(self, U=0.2, tau_f=1500, tau_d=200):
        """
        Initialize the STP model parameters.
        U: Utilization of synaptic efficacy
        tau_f: Time constant of facilitation (ms)
        tau_d: Time constant of depression (ms)
        """
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.u = 0
        self.x = 1

    def update(self, dt, spike):
        """
        Update the synaptic variables based on the spike input.
        This is for excitatory-to-excitatory synapses. 
        dt: Time step (ms)
        spike: Binary spike input (0 or 1)
        """
        dx = (1-self.x)/self.tau_d - self.u*self.x*spike
        du = (self.U-self.u)/self.tau_f + self.U*(1-self.u)*spike

        self.x += dx*dt
        self.u += du*dt

        return self.u*self.x

# Example usage
if __name__ == "__main__":
    stp = STPModel()
    dt = 1
    spikes = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]

    for spike in spikes:
        efficacy = stp.update(dt, spike)
        print(f"Spike: {spike}, Synaptic Efficacy: {efficacy}")
