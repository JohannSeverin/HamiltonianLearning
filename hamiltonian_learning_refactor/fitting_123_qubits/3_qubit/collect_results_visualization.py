path_data = "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits"
path_simulated = "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/3_qubit/fitting_scripts"

%matplotlib widget 
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown
from IPython.display import display
import xarray as xr
import numpy as np 
import pickle

plt.style.use("/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/utils/presentation_figure_template.mplstyle")


losses = {}


number_of_params = {
    "H1L1": 3 * 3 + 3 * 9,
    "H2L1": 3 * 3 + 3 * 9  + 3 * 9,
    "H3L1": 3 * 3 + 3 * 9 + 27  + 3 * 9,
    "H1L2": 3 * 3 + 3 * 225 - 3 * 9,
    "H2L2": 3 * 3 + 3 * 9 + 3 * 9 + 3 * (225 - 9 - 9),
    "H3L2": 3 * 3 + 3 * 9 + 27 + 3 * 9 + 3 * (225 - 9- 9),
    "H1L3": 3 * 3 + (4 ** 3 - 1) ** 2,
    "H2L3": 3 * 3 + 3 * 9 + (4 ** 3 - 1) ** 2,
    "H3L3": 3 * 3 + 3 * 9 + 27 + (4 ** 3 - 1) ** 2,
}

items_to_get = list(number_of_params.keys()) #  ["H1L1", "H2L1", "H1L2", "H3L1", "H2L2",  "H2L2", "H3L2", "H2L3", "H3L3"]

for item in items_to_get:
    with open(path_simulated + f"/parameters_{item}.pickle", "rb") as f:
        losses[item] = pickle.load(f)["final_loss"]

colors = {
    "H1L1": "C0",
    "H2L1": "C1",
    "H1L2": "C7",
    "H3L1": "C2",
    "H2L2": "C3",
    "H3L2": "C4",
    "H1L3": "C8",
    "H2L3": "C5",
    "H3L3": "C6",
}

### 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax_nllh = axes[0]

h, l = np.arange(1, 4), np.arange(1, 4)
hh, ll = np.meshgrid(l, h)

log_likelihoods = np.zeros(ll.shape)

for i in h:
    for j in l:
        key = f"H{i}L{j}"
        log_likelihoods[j-1, i -1] = losses[key]


number_of_parameters = np.zeros(ll.shape)

for i in h:
    for j in l:
        key = f"H{i}L{j}"
        number_of_parameters[j-1, i-1] = number_of_params[key]
        

# log_likelihoods[0, 0] = np.nan

L_tick_labels = [r"$\theta_{L" + str(i) + r"}$" for i in range(1, 4)]
H_tick_labels = [r"$\theta_{H" + str(i) + r"}$" for i in range(1, 4)]

im_values = log_likelihoods
# im_values[0, 0] = np.nan

cmap = plt.cm.get_cmap("Blues_r")
cmap.set_bad(color="gray")
cmap.set_over(color="gray")

im = ax_nllh.imshow(im_values, cmap=cmap, origin="lower", vmax = np.sort(log_likelihoods.flatten())[-2])
ax_nllh.set_xticks(np.arange(0, 3, 1))
ax_nllh.set_yticks(np.arange(0, 3, 1))
ax_nllh.set_yticklabels(L_tick_labels)
ax_nllh.set_xticklabels(H_tick_labels)

from matplotlib import patheffects

ax_nllh.vlines(np.arange(len(h)) + 0.5, -0.5, len(l)-0.5, colors='black', linestyles='-', linewidths=0.5)
ax_nllh.hlines(np.arange(len(l)) + 0.5, -0.5, len(h)-0.5, colors='black', linestyles='-', linewidths=0.5)
for j in range(len(h)):
    for i in range(len(l)):
        color = "white" # black" if i == 0 and j == 0 else "white"
        text = ax_nllh.text(j, i, f"${np.round(1e-6 * log_likelihoods[i, j], 2)}$", ha="center", va="center", color=color, path_effects=[patheffects.withStroke(linewidth=1, foreground='black')], weight="bold")

cax = ax_nllh.inset_axes([-0.20, 0.00, 0.05, 1.00])
cbar = plt.colorbar(im, cax=cax, label="Log Likelihood $(10^{6})$", extend = "max", extendrect=True)
cax.yaxis.set_label_position("left")
cax.yaxis.set_ticks_position("left")
cbar.set_ticklabels(["", "", "", "", "", ""])

ax_EP_values = axes[1]

ax_EP_values.scatter(np.ravel(hh), np.ravel(ll), s = 2000, marker = "s", c = np.ravel(im_values), cmap=cmap, edgecolor="black", linewidth=0.5, vmax = np.sort(log_likelihoods.flatten())[-2])
# ax_EP_values.scatter(ll[0,0], hh[0,0], s = 2000, marker = "s", c = "gray", edgecolor="black", linewidth=0.5)




# Draw arrows to indicate the direction of the EP 
def calculate_explanatory_power(difference_in_log_likelihood, difference_in_parameters):
    return (2 * difference_in_log_likelihood - difference_in_parameters) / np.sqrt(2 * difference_in_parameters)


from matplotlib.patches import FancyArrowPatch

offsets = 0.25
text_offsets = 0.12


# Calculate the explanatory power values
EP_values_right = np.log10(np.abs(calculate_explanatory_power(log_likelihoods[:, :-1] - log_likelihoods[:, 1:], number_of_parameters[:, 1:] - number_of_parameters[:, :-1])))
EP_values_up = np.log10(np.abs(calculate_explanatory_power(log_likelihoods[:-1, :] - log_likelihoods[1:, :], number_of_parameters[1:, :] - number_of_parameters[:-1, :])))
EP_values_diagonal = np.log10(np.abs(calculate_explanatory_power(log_likelihoods[:-1, :-1] - log_likelihoods[1:, 1:], number_of_parameters[1:, 1:] - number_of_parameters[:-1, :-1])))

min_EP = min(EP_values_right.min(), EP_values_up.min(), EP_values_diagonal.min())
max_EP = max(EP_values_right.max(), EP_values_up.max(), EP_values_diagonal.max())

EP_values_right = calculate_explanatory_power(log_likelihoods[:, :-1] - log_likelihoods[:, 1:], number_of_parameters[:, 1:] - number_of_parameters[:, :-1])
EP_values_up = calculate_explanatory_power(log_likelihoods[:-1, :] - log_likelihoods[1:, :], number_of_parameters[1:, :] - number_of_parameters[:-1, :])
EP_values_diagonal = calculate_explanatory_power(log_likelihoods[:-1, :-1] - log_likelihoods[1:, 1:], number_of_parameters[1:, 1:] - number_of_parameters[:-1, :-1])


# min_EP, max_EP
cmap = plt.cm.get_cmap("gist_heat_r")

def format_EP(value):
    sign = "+" if value > 0 else "-"
    order = np.floor(np.log10(np.abs(value)))
    if order < 2:
        return f"${int(np.round(value))}$"
    else:
        return f"${int(np.round(value / 10 ** order))} \\times 10^{{{int(order)}}}$"
        # Draw fancy arrows to the right
for j in range(len(h)-1):
    for i in range(len(l)):
        color = cmap((np.log10(np.abs(EP_values_right[i, j])) - min_EP) / (max_EP - min_EP))
        ax_EP_values.add_patch(FancyArrowPatch((hh[i, j] + offsets, ll[i, j]), (hh[i, j+1] - offsets, ll[i, j]), arrowstyle='simple', fc=color, ec=color, mutation_scale=25, path_effects=[patheffects.withStroke(linewidth=2, foreground='black')]))
        text = ax_EP_values.text((hh[i, j] + hh[i, j+1]) / 2, ll[i, j] + text_offsets, format_EP(EP_values_right[i, j]), ha="center", va="center", color="black", fontsize=10)

# Draw fancy arrows upwards
for j in range(len(h)):
    for i in range(len(l)-1):
        color = cmap((np.log10(np.abs(EP_values_up[i, j])) - min_EP) / (max_EP - min_EP))
        ax_EP_values.add_patch(FancyArrowPatch((hh[i, j], ll[i, j] + offsets), (hh[i, j], ll[i+1, j] - offsets), arrowstyle='simple', fc=color, ec=color, mutation_scale=25, path_effects=[patheffects.withStroke(linewidth=2, foreground='black')]))
        text = ax_EP_values.text(hh[i, j] - text_offsets, (ll[i, j] + ll[i+1, j]) / 2, format_EP(EP_values_up[i, j]), ha="center", va="center", color="black", fontsize=10, rotation = 90)	

# # And diagonally
# for i in range(len(l)-1):
#     for j in range(len(h)-1):
#         color = cmap((np.log10(np.abs(EP_values_diagonal[i, j])) - min_EP) / (max_EP - min_EP))
#         ax_EP_values.add_patch(FancyArrowPatch((ll[i, j] + offsets, hh[i, j] + offsets), (ll[i+1, j + 1] - 1.4 * offsets, hh[i+1, j + 1] - 1.4 * offsets), arrowstyle='simple', fc=color, ec=color, mutation_scale=25))
#         text = ax_EP_values.text((ll[i, j] + ll[i+1, j+1]) / 2 - text_offsets, (hh[i, j] + hh[i+1, j+1]) / 2 + text_offsets, format_EP(EP_values_diagonal[i, j]), ha="center", va="center", color="black", fontsize=10, rotation=45)


ax_EP_values.set_xticks(np.arange(1, 4))
ax_EP_values.set_yticks(np.arange(1, 4))

ax_EP_values.set_yticklabels(L_tick_labels)
ax_EP_values.set_xticklabels(H_tick_labels)

ax_EP_values.set(xlim=(0.5, 3.5), ylim=(0.5, 3.5))

ax_nllh.set(title="Log Likelihoods")

# Remove ax lines and make ticks invisible
ax_EP_values.spines['top'].set_visible(True)
ax_EP_values.spines['right'].set_visible(True)
ax_EP_values.spines['bottom'].set_visible(True)
ax_EP_values.spines['left'].set_visible(True)
# ax_EP_values.tick_params(axis='both', which='both', length=0)
ax_EP_values.set(title = "Explanatory Power")



from matplotlib.cm import ScalarMappable

cax = ax_EP_values.inset_axes([1.10, 0.00, 0.05, 1.00])
cbar = plt.colorbar(ScalarMappable(cmap=cmap), cax=cax, label="Log Explanatory Power", extend = "max", extendrect=True)
cbar.set_ticklabels(["", "", "", "", "", ""])
cax.yaxis.set_label_position("left")


fig.tight_layout()
fig.savefig("3_qubit_log_likelihoods_and_explanatory_power.pdf")

# Test 3d bar chart 


        
# fig = plt.figure(figsize=(10, 5))
# ax_nllh = fig.add_subplot(111, projection='3d')

# x = np.arange(1, 4)
# y = np.arange(1, 4)
# x, y = np.meshgrid(x, y)

# log_likelihoods[0, 0] = np.nan 

# ax_nllh.bar3d(x.ravel(), y.ravel(), np.zeros_like(log_likelihoods).ravel(), 0.8, 0.8, np.log(log_likelihoods).ravel(), cmap="viridis")

# ax_nllh.set_xlabel('H')
# ax_nllh.set_ylabel('L')
# ax_nllh.set_zlabel('Log Likelihood')

# plt.xticks(np.arange(1, 4))
# plt.yticks(np.arange(1, 4))

# plt.show()