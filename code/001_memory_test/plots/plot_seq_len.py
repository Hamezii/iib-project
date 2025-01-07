import numpy as np
import matplotlib.pyplot as plt

x_32 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]
y_32 = [99.9, 99.9, 99.9, 99.9, 99.8, 99.6, 99.13, 92.46, 83.025, 86.95, 81.54, 84.48, 61.33, 57.74]
x_16 = [2, 3, 4, 5, 6, 7, 8, 9]
y_16 = [99.9, 99.9, 99.63, 86.0, 77.62, 65.64, 61.09, 59.33]

plt.plot(x_32, y_32, label="$H=32$")
plt.plot(x_16, y_16, label="$H=16$")


plt.ylim(45, 100)
plt.xlim(0, 17)
plt.xlabel("Sequence Length to Memorize")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Sequence Length")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_seq_length.svg")
plt.show()