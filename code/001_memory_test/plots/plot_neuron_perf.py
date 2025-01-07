import numpy as np
import matplotlib.pyplot as plt

x = [8, 12, 16, 24, 32, 48, 64, 80]
y = [48.20, 55.27, 60.78, 77.81, 85.00 , 89.58, 92.61, 97.99]

plt.plot(x, y)


plt.ylim(45, 100)
plt.xlim(0, 80)
plt.xlabel("Neurons")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Neurons For Sequences of Length 2-5")
#plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_neurons.svg")
plt.show()