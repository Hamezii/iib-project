import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7]
y128 = [23, 100, 100, 99.55, 97.58, 28.08, 15.03]
y96 = [70.3, 100.0, 99.96666666666667, 99.7, 94.72, 25.05, 10.7]
y64 = [41.1, 99.85, 98.56666666666666, 93.9, 85.06, 56.583333333333336, 29.67142857142857]
y32 = [76.4, 99.7, 94.7, 84.65, 68.82, 47.916666666666664, 30.285714285714285]


plt.plot(x, y128, label="$H=128$", linewidth=2, color='red')
plt.plot(x, y96, label="$H=96$", linewidth=2, color='orange')
plt.plot(x, y64, label="$H=64$", linewidth=2, color='green')
plt.plot(x, y32, label="$H=32$", linewidth=2, color='blue')



plt.ylim(0, 100)
plt.xlim(0, 7)
plt.xlabel("Sequence Length")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Sequence Length trained with N=[2, 5]")
plt.grid(True)
plt.legend()
plt.savefig("accuracy_vs_seq_length_single_model.svg")
plt.show()