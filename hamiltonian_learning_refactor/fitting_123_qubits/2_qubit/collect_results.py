path_data = "/root/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/2_qubit/"
path_simulated = "/root/projects/HamiltonianLearning/hamiltonian_learning_refactor/fitting_123_qubits/2_qubit/fitting_scripts/"

%matplotlib widget 
import matplotlib.pyplot as plt
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
    "H1L1": 24,
    "H2L1": 15 + 9 + 9,
    "H2L2": 15 + 225
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Plot losses as a function of number of parameters
x = np.array(list(number_of_params.values()))
y = 1e-6 *  np.array(list(losses.values()))

ax.plot(x, y, 'o')
ax.set_xlabel("Number of Parameters")
ax.set_ylabel("Negative Log Likelihood (${10}^6$)")
ax.set_title("Losses vs Number of Parameters")

# Add labels for each model
for i, item in enumerate(items_to_get):
    ax.annotate(item, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

ax.set(
    xlim=(-10, 250),
    ylim = (0, 5)
)

# Show the plot
plt.show()



# Subset of the data to analyze the difference
subset = ["H2L1", "H2L2"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]}, sharey = True)

# Plot losses as a function of number of parameters
x = np.array([number_of_params[item] for item in subset])
y = np.array([losses[item] for item in subset])

for i, item in enumerate(subset):
    axes[1].plot(x[i], y[i], 'o', label=item)
    axes[1].annotate(item, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

axes[1].set_xlabel("Number of Parameters")
axes[1].set_ylabel("Negative Log Likelihood")
axes[1].set_title("Losses vs Number of Parameters")


# Chi2 distribution
from scipy.stats import chi2 

x = np.linspace(0, 200, 1000)




y = chi2.pdf(x * 2, df=number_of_params["H2L2"] - number_of_params["H2L1"])
axes[0].plot(y, losses[subset[0]] - x, label="Chi2 distribution")
