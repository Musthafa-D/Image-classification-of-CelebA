def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

import torch
import matplotlib.pyplot as plt
#from tueplots import figsizes
import matplotlib.ticker as ticker

# Load the data storage file
data_storage = torch.load('//media//NAS//400__Server//Mohideen//celeba_classification//00_Runs//2024-08-17_10-08-30//data_storage.pt')
# data_storage = torch.load('C://Users//DiwanMohideen//sciebo//00_gitlab//mnist_classification//00_Runs//2024-07-07_13-26-13//data_storage.pt')

# Example access to specific stored values
batches = data_storage.stored_values["batch"]

train_accuracies = data_storage.get_item("train_acc")
test_accuracies = data_storage.get_item("test_acc")
average_train_accuracies = data_storage.get_item("a_train_acc")

train_loss = data_storage.get_item("train_loss")
test_loss = data_storage.get_item("test_loss")
average_train_loss = data_storage.get_item("a_train_loss")

learning_rate = data_storage.get_item("learning_rate")
learning_rate_l = data_storage.get_item("learning_rate_l")

# Print the values or perform further analysis
#print("Batches:", batches)
#print("Train Accuracies:", train_accuracies)
#print("Test Accuracies:", test_accuracies)
#print("Average Train Accuracies:", average_train_accuracies)
print("learning_rate:", learning_rate)
print("learning_rate_l:", learning_rate_l)

"""Accuracies"""
x = batches
ytr = train_accuracies
yatr = average_train_accuracies
yt = test_accuracies

figs = []
names = []

plt.rcParams.update({"figure.dpi": 300})
# plt.rcParams.update(figsizes.icml2022_half())
# icml_size = figsizes.icml2022_half()
# print(icml_size)
# fig, ax = plt.subplots()

fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
print(set_size('thesis'))

ax.plot(x, ytr, label='$\\mathrm{Acc}_{\\mathrm{train}}$')
ax.plot(x, yatr, label='$\\mathrm{Acc}_{\\mathrm{train\\_avg}}$')
ax.plot(x, yt, label='$\\mathrm{Acc}_{\\mathrm{test}}$')

ax.set_xlabel("$n$", fontsize=14)
ax.set_ylabel("$\\mathrm{Acc}$", fontsize=14)

ax.set_xlim(left=0, right=max(x))
ax.set_ylim(bottom=0, top=100)

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

ax.grid(True)
ax.set_yticks(range(10, 101, 10))
ax.legend()

figs.append(fig)
plt.savefig('accuracies.pdf')
plt.show()
plt.close(fig)

"""Losses"""
x = batches
ytr = train_loss
yatr = average_train_loss
yt = test_loss

figs = []
names = []

plt.rcParams.update({"figure.dpi": 300})
# plt.rcParams.update(figsizes.icml2022_half())
# icml_size = figsizes.icml2022_half()
# print(icml_size)
# fig, ax = plt.subplots()

fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
print(set_size('thesis'))

ax.plot(x, ytr, label='$\\mathcal{L}_{\\mathrm{train}}$')
ax.plot(x, yatr, label='$\\mathcal{L}_{\\mathrm{train\\_avg}}$')
ax.plot(x, yt, label='$\\mathcal{L}_{\\mathrm{test}}$')

ax.set_xlabel("$n$", fontsize=14)
ax.set_ylabel('$\\mathcal{L}$', fontsize=14)

ax.set_xlim(left=0, right=max(x))
ax.set_ylim(bottom=0, top=max(ytr))

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

ax.grid(True)
ax.set_yticks(range(1, 3, 1))
ax.legend()

figs.append(fig)
plt.savefig('losses.pdf')
plt.show()
plt.close(fig)


"""LR Schedule"""
figs = []
names = []

plt.rcParams.update({"figure.dpi": 300})
# plt.rcParams.update(figsizes.icml2022_half())
# icml_size = figsizes.icml2022_half()
# print(icml_size)
# fig, ax = plt.subplots()

fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
print(set_size('thesis'))

ax.plot(learning_rate, label="$\\mathrm{lr}_{\\mathrm{m}}$")
ax.plot(learning_rate_l, label="$\\mathrm{lr}_{\\mathrm{f}}$")

ax.set_xlabel('$n$', fontsize=14)
ax.set_ylabel('$lr$', fontsize=14)

ax.set_xlim(left=0, right=max(x))
ax.set_ylim(bottom=0)

# Use ScalarFormatter to display y-axis in scientific notation (powers of 10)
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

ax.legend()

figs.append(fig)
plt.savefig('learning_rate_schedule.pdf')
plt.show()
plt.close(fig)
