# lambda 0 -> 1.00, 0.0302
# lambda 2 -> 0.66, 0.0281
# lambda 4 -> 0.48, 0.0288
# lambda 6 -> 0.38, 0.0268
# lambda 8 -> 0.29, 0.0266
# lambda 10 -> 0.24, 0.0266

import matplotlib.pyplot as plt

# Data points
lambda_values = [0, 2, 4, 6, 8, 10]
sparsity_ratios = [1.00, 0.66, 0.48, 0.38, 0.29, 0.24]
mse_values = [0.0302, 0.0281, 0.0288, 0.0268, 0.0266, 0.0266]

# Create a plot with two y-axes
fig, ax1 = plt.subplots()

# Plot MSE on the first y-axis
ax1.set_xlabel("$\lambda$")
ax1.set_ylabel("MSE")
line1 = ax1.scatter(lambda_values, mse_values, color="tab:blue", marker="o", label="MSE")
ax1.tick_params(axis="y")
ax1.set_ylim(0.025, 0.032)

# Create a second y-axis to plot sparsity ratio
ax2 = ax1.twinx()
ax2.set_ylabel("Sparsity Ratio")
line2 = ax2.scatter(lambda_values, sparsity_ratios, color="tab:orange", marker="s", label="Sparsity Ratio")
ax2.tick_params(axis="y")
ax2.set_ylim(0, 1.1)


lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper right")
# ax2.legend([line2], ["Sparsity Ratio"], loc="upper right")

# Title and layout
fig.tight_layout()

plt.show()
