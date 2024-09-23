path_data = "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits"
path_simulated = "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/3_qubit/fitting_scripts"

%matplotlib widget 
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown
from IPython.display import display
import xarray as xr
import numpy as np 
import pickle

items_to_get = ["H1L1","H2L1", "H3L1"] # ,  "H2L2", "H3L2", "H2L3", "H3L3"]


losses = {}

for item in items_to_get:
    with open(path_simulated + f"/parameters_{item}.pickle", "rb") as f:
        losses[item] = pickle.load(f)["final_loss"]


number_of_params = {
    "H1L1": 3 * 3 + 3 * 9,
    "H2L1": 3 * 3 + 3 * 9  + 3 * 9,
    "H3L1": 3 * 3 + 3 * 9 + 27  + 3 * 9,
    # "H2L2": 3 * 3 + 3 * 9 + 3 * 9 + 3 * (225 - 9),
    # "H3L2": 3 * 3 + 3 * 9 + 27 + 3 * 9 + 3 * (225 - 9),
    # "H2L3": 3 * 3 + 3 * 9 + (4 ** 3 - 1) ** 2,
    # "H3L3": 3 * 3 + 3 * 9 + 27 + (4 ** 3 - 1) ** 2,
}


colors = {
    "H1L1": "C0",
    "H2L1": "C1",
    "H3L1": "C2",
    "H2L2": "C3",
    "H3L2": "C4",
    "H2L3": "C5",
    "H3L3": "C6",
}


# from matplotlib.gridspec import GridSpec

# gs = GridSpec(2, 3)

# fig = plt.figure(figsize=(8, 8))

# ax_full = fig.add_subplot(gs[0, :])

# # Plot losses as a function of number of parameters
# names = list(number_of_params.keys())
# loss_values = np.array(list(losses.values()))
# index = np.arange(len(names))

# loss_values_scaled = loss_values #  / loss_values.max()

# for i, item in enumerate(names):
# ax_full.scatter(i, loss_values_scaled[i], marker  ='o', label=item, s = 100)
# # ax_full.annotate(item, (i, loss_values[i]), textcoords="offset points", xytext=(0,10), ha='center')


# ticks = [r"$\Theta_{" + f"{names[i]}" + r"}$" for i in range(len(names))]
# number_of_params_tick = np.array([number_of_params[name] for name in names])

# ax_full.set_xticks(index)
# ax_full.set_xticklabels(number_of_params_tick)

# ax_full_upper = ax_full.twiny()
# ax_full_upper.set_xlim(ax_full.get_xlim())
# ax_full_upper.set_xticks(index)
# ax_full_upper.set_xticklabels(ticks)

# ax_full.set_xlabel("Number of Parameters")



# nllh = "$-\log(\mathbb{L})$"
# ax_full.set_ylabel(f"{nllh} / max({nllh})")
# # ax_full.set_yscale("log")

# fig.tight_layout()



### 

from matplotlib.gridspec import GridSpec

gs = GridSpec(2, 3)

fig = plt.figure(figsize=(8, 8))

ax_full = fig.add_subplot(gs[0, :])

# Plot losses as a function of number of parameters
names = list(number_of_params.keys())
loss_values = np.array(list(losses.values()))
index = np.arange(len(names))

loss_values_scaled = loss_values / loss_values.max()

for i, item in enumerate(names):
    ax_full.scatter(i, loss_values_scaled[i], marker  ='o', label=item, s = 100, color = colors[item])
    # ax_full.annotate(item, (i, loss_values[i]), textcoords="offset points", xytext=(0,10), ha='center')


ticks = [r"$\Theta_{" + f"{names[i]}" + r"}$" for i in range(len(names))]
number_of_params_tick = np.array([number_of_params[name] for name in names])

ax_full.set_xticks(index)
ax_full.set_xticklabels(number_of_params_tick)

ax_full_upper = ax_full.twiny()
ax_full_upper.set_xlim(ax_full.get_xlim())
ax_full_upper.set_xticks(index)
ax_full_upper.set_xticklabels(ticks)

ax_full.set_xlabel("Number of Parameters")



nllh = "$-\log(\mathbb{L})$"
ax_full.set_ylabel(f"{nllh} / max({nllh})")
ax_full.set_yscale("log")
# ax_full.set_ylim(1e-2, 2)


ax_zoomin = fig.add_subplot(gs[1, :-1])


name_zoomin = names[-2:]
loss_zoomin = loss_values[-2:]
params_zoomin = np.array([number_of_params[name] for name in name_zoomin])
index_zoomin = index[-2:]

for i, item in enumerate(name_zoomin):
    ax_zoomin.scatter(params_zoomin[i], loss_zoomin[i], marker  ='o', label=item, s = 100, color = colors[item], zorder = 10)
    # ax_zoomin.annotate(item, (params_zoomin[i], loss_zoomin[i]), textcoords="offset points", xytext=(0,10), ha='center')



# The gauss approximation 
x_params = np.arange(params_zoomin.min() + 1, params_zoomin.max() + 10)
diff_params = x_params - params_zoomin[0]

mean_expected_loss = loss_zoomin[0] - .5 * (diff_params)  
std_expected_loss = np.sqrt(2 * diff_params)

ax_zoomin.plot(x_params, mean_expected_loss, label="Expected Loss", color="black")
ax_zoomin.plot(x_params, mean_expected_loss + std_expected_loss, label="Expected Loss + std", color="black", linestyle="--")
ax_zoomin.plot(x_params, mean_expected_loss - std_expected_loss, label="Expected Loss - std", color="black", linestyle="--")


ticks = [r"$\Theta_{" + f"{name}" + r"}$" for name in name_zoomin]
params_tick = [number_of_params[name] for name in name_zoomin]

ax_zoomin.set_xticks(params_zoomin)
ax_zoomin.set_xticklabels(params_tick)

ax_zoomin.set_xlabel("Number of Parameters")


ax_zoomin_upper = ax_zoomin.twiny()
ax_zoomin_upper.set_xlim(ax_zoomin.get_xlim())
ax_zoomin_upper.set_xticks(params_zoomin)
ax_zoomin_upper.set_xticklabels(ticks)




# # Make the imshow plot 
y_loss = np.linspace(loss_zoomin.min(), loss_zoomin.max(), 1000)
lambda_test_statistic = 2 * (loss_zoomin[0] - y_loss)

from scipy.stats import chi2
xx, yy = np.meshgrid(x_params, y_loss)

chi2_values = chi2.pdf(lambda_test_statistic, df = diff_params[:, None])

chi2_values = chi2_values / chi2_values.max(axis=1)[:, None]

# ax_zoomin.imshow(chi2_values, aspect="auto", extent=[x_params.min(), x_params.max(), y_loss.min(), y_loss.max()], cmap="Reds", origin = "lower")


from scipy.stats import chi2


lambdas_to_test = 2 * np.linspace(0.1, 3.0 * np.max(diff_params), 1000)

chi2_values = chi2.pdf(lambdas_to_test, df = diff_params[:, None])
chi2_values_scaled = chi2_values / chi2_values.max(axis=1)[:, None]

from matplotlib.colors import LinearSegmentedColormap

colormap_from_white_to_red = LinearSegmentedColormap.from_list("red", [(1, 1, 1), (1, 0, 0)], N=256)


ax_zoomin.pcolor(x_params, y_loss.max() - 0.5 * lambdas_to_test, chi2_values_scaled.T, cmap=colormap_from_white_to_red, shading="nearest")

ax_zoomin.set_ylabel(f"{nllh}")
# ax_zoomin.autoscale(False)

# ax_zoomin.vlines(params_zoomin[-1], *ax_zoomin.get_ylim() , color="red", linestyle="-")



# # Do the line cut at the last point 
ax_linecut = fig.add_subplot(gs[1, -1])

ax_linecut.plot(chi2_values[-1, :], y_loss.max() - 0.5 * lambdas_to_test, color="red")

chi2_statistic_for_measurement = chi2.pdf(2 * (loss_zoomin[0] - loss_zoomin[-1]), df = diff_params[-1])
ax_linecut.scatter(chi2_statistic_for_measurement, loss_zoomin[-1], color="black", s = 100, zorder = 10)

ax_linecut.set(
    xlabel = "PDF",
    xlim = (-0.005, 0.020)
)

ax_linecut.yaxis.set_visible(False)

ax_linecut.sharey(ax_zoomin)


fig.tight_layout()




# ax_full.set_ylim(1e-2, 2)


# ax_zoomin = fig.add_subplot(gs[1, :-1])



# fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# # Plot losses as a function of number of parameters
# x = np.array(list(number_of_params.values()))
# y = 1e-6 *  np.array(list(losses.values()))

# ax.plot(x, y, 'o')
# ax.set_xlabel("Number of Parameters")
# ax.set_ylabel("Negative Log Likelihood (${10}^6$)")
# ax.set_title("Losses vs Number of Parameters")

# # ax.set_yscale("log")
# ax.set_xscale("log")


# # Add labels for each model
# for i, item in enumerate(items_to_get):
#     ax.annotate(item, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

# # ax.set(
# #     xlim=(-10, 250),
# #     ylim = (0, 5)
# # )

# # Show the plot
# plt.show()



# # Subset of the data to analyze the difference
# subset = ["H2L1", "H2L2"]

# fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]}, sharey = True)

# # Plot losses as a function of number of parameters
# x = np.array([number_of_params[item] for item in subset])
# y = np.array([losses[item] for item in subset])

# for i, item in enumerate(subset):
#     axes[1].plot(x[i], y[i], 'o', label=item)
#     axes[1].annotate(item, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

# axes[1].set_xlabel("Number of Parameters")
# axes[1].set_ylabel("Negative Log Likelihood")
# axes[1].set_title("Losses vs Number of Parameters")


# # Chi2 distribution
# from scipy.stats import chi2 

# x = np.linspace(0, 200, 1000)




# y = chi2.pdf(x * 2, df=number_of_params["H2L2"] - number_of_params["H2L1"])
# axes[0].plot(y, losses[subset[0]] - x, label="Chi2 distribution")
