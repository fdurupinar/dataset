import matplotlib.pyplot as plt
import numpy as np

# Data from the table
traits = ['Open.', 'Consc.', 'Extro.', 'Agree.', 'Stab.', 'Avg.']
test_acc_iter_1_cnn = [83.50, 83.95, 85.60, 84.32, 83.90, 84.10]
test_acc_iter_2_cnn = [90.37, 88.25, 88.23, 90.06, 89.86, 89.35]
test_acc_iter_1_hybrid = [84.70, 83.60, 84.58, 84.70, 83.25, 83.90]
test_acc_iter_2_hybrid = [90.75, 89.90, 86.28, 91.80, 91.29, 89.99]

# Colors for each trait
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
plt.rcParams.update({'font.size': 24})

# X positions for the traits
x = np.arange(len(traits))

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)
width = 0.2  # Width of the bars
# CNN Model
for i in range(len(traits)):
    axes[0].plot([x[i], x[i]], [test_acc_iter_1_cnn[i], test_acc_iter_2_cnn[i]], marker='o', color=colors[i],markersize=10, linewidth=30, label=traits[i] if i == 0 else "")
    # axes[0].bar(x[i] - width, test_acc_iter_1_cnn[i], width, label='CNN Iter. 1', color='blue')
    # axes[0].bar(x[i] - test_acc_iter_2_cnn[i], width, label='CNN Iter. 1', color='blue')

    # bars1 = ax.bar(x - width, test_acc_iter_1_cnn, width, label='CNN Iter. 1', color='blue')
    # bars2 = ax.bar(x, test_acc_iter_2_cnn, width, label='CNN Iter. 2', color='orange')


axes[0].set_xticks(x)
axes[0].set_xticklabels(traits, rotation=45, ha='right')

axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('CNN Model')

# Hybrid Model
for i in range(len(traits)):
    axes[1].plot([x[i], x[i]], [test_acc_iter_1_hybrid[i], test_acc_iter_2_hybrid[i]], marker='o', color=colors[i], markersize=10, linewidth=30,label=traits[i] if i == 0 else "")

axes[1].set_xticks(x)
axes[1].set_xticklabels(traits, rotation=45, ha='right')

axes[1].set_title('Hybrid Model')


# Adding grid
for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()
