path_data = "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/2_qubit/"
path_simulated = "/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/2_qubit/fitting_scripts/"

%matplotlib widget 
import matplotlib.pyplot as plt

plt.style.use("/home/archi1/projects/HamiltonianLearning/hamiltonian_learning_refactor/utils/presentation_figure_template.mplstyle")

from ipywidgets import interact, Dropdown
from IPython.display import display
import xarray as xr
import numpy as np 
import pickle

items_to_get = ["H1L0", "H1L1", "H2L0", "H2L1", "H2L2"]


losses = {}

for item in items_to_get:
    with open(path_simulated + f"parameters_{item}.pickle", "rb") as f:
        losses[item] = pickle.load(f)["final_loss"]


number_of_params = {
    "H1L0": 6,
    "H2L0": 15,
    "H1L1": 3 + 3 + 9 + 9,
    "H2L1": 15 + 9 + 9,
    "H2L2": 15 + 225
}

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
    ax_full.scatter(i, loss_values_scaled[i], marker  ='o', label=item, s = 100)
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
ax_full.set_ylim(1e-2, 2)


ax_zoomin = fig.add_subplot(gs[1, :-1])


name_zoomin = names[-2:]
loss_zoomin = loss_values[-2:]
params_zoomin = np.array([number_of_params[name] for name in name_zoomin])
index_zoomin = index[-2:]

for i, item in enumerate(name_zoomin):
    ax_zoomin.scatter(params_zoomin[i], loss_zoomin[i], marker  ='o', label=item, s = 100, color = f"C{index_zoomin[i]}", zorder = 10)
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




# Make the imshow plot 
y_loss = np.linspace(loss_zoomin.min(), loss_zoomin.max(), 1000)
lambda_test_statistic = 2 * (loss_zoomin[0] - y_loss)

from scipy.stats import chi2
xx, yy = np.meshgrid(x_params, y_loss)

chi2_values = chi2.pdf(lambda_test_statistic, df = diff_params[:, None])

chi2_values = chi2_values / chi2_values.max(axis=1)[:, None]

# ax_zoomin.imshow(chi2_values, aspect="auto", extent=[x_params.min(), x_params.max(), y_loss.min(), y_loss.max()], cmap="Reds", origin = "lower")


from scipy.stats import chi2


lambdas_to_test = 2 * np.linspace(0.1, 0.60 * np.max(diff_params), 1000)

chi2_values = chi2.pdf(lambdas_to_test, df = diff_params[:, None])
chi2_values_scaled = chi2_values / chi2_values.max(axis=1)[:, None]

from matplotlib.colors import LinearSegmentedColormap

colormap_from_white_to_red = LinearSegmentedColormap.from_list("red", [(1, 1, 1), (1, 0, 0)], N=256)


ax_zoomin.pcolor(x_params, y_loss.max() - 0.5 * lambdas_to_test, chi2_values_scaled.T, cmap=colormap_from_white_to_red, shading="nearest")

ax_zoomin.set_ylabel(f"{nllh}")
# ax_zoomin.autoscale(False)

ax_zoomin.vlines(params_zoomin[-1], *ax_zoomin.get_ylim() , color="red", linestyle="-")



# Do the line cut at the last point 
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
fig.savefig("two_qubit_loss_vs_parameters.pdf")

# 

