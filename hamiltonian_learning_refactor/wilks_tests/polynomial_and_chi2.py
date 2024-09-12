import iminuit
import numpy as np
import matplotlib.pyplot as plt

from iminuit.cost import LeastSquares


def fit_polynomial(xdata, ydata, yerr, degree, with_const=True):
    """
    Fit a polynomial to data
    """

    def func(x):
        return np.sum((np.polyval(x, xdata) - ydata) ** 2 / yerr**2)

    args_guess = np.polyfit(xdata, ydata, degree)

    if not with_const:
        args_guess[-1] = 0

    m = iminuit.Minuit(func, args_guess)

    args_guess = np.polyfit(xdata, ydata, degree)

    if not with_const:
        m.fixed[-1] = True

    m.migrad()

    return m


# Data
xdata = np.arange(0, 10, 1)

np.random.seed(0)
ydata = 3 + np.random.normal(0, 1, 10)
yerr = np.ones(10)


# Plotting

# Figure
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))


ax = axes[0]
ax.scatter(xdata, ydata, label="Data", color="red")
ax.errorbar(
    xdata,
    ydata,
    yerr,
    marker="None",
    linestyle="None",
    capsize=3,
    elinewidth=1,
    color="black",
)
ax.set(xlabel="x", ylabel="y", title="Polynomial fit", ylim=(-1, 7))

# Fit a polynomial

chi_squares = []

max_degress = 9
add_const_after_degree = 9

with_const = [False] * add_const_after_degree + [True] * (
    max_degress - add_const_after_degree
)

cmapper = plt.get_cmap("viridis")

for i in range(1, max_degress):
    m = fit_polynomial(xdata, ydata, yerr, i, with_const=with_const[i - 1])
    ax.plot(
        np.linspace(xdata.min(), xdata.max(), 1000),
        np.polyval(
            m.values,
            np.linspace(xdata.min(), xdata.max(), 1000),
        ),
        label=f"Degree {i} fit",
        color=cmapper(i / max_degress),
    )

    axes[1].scatter(
        i,
        m.fval,
        color=cmapper(i / max_degress),
    )

    chi_squares.append(m.fval)

    # Save figure
    fig.savefig(f"figs/polynomial_fit_{i}.png")


ax_insert = axes[1].inset_axes([0.5, 0.5, 0.5, 0.3])

ax_insert.scatter(
    range(3, max_degress),
    chi_squares[2:],
    marker="o",
    linestyle="None",
    c=cmapper(np.arange(3, max_degress) / max_degress),
)
ax_insert.set(ylim=(np.min(chi_squares[2:]) - 1, np.max(chi_squares[2:]) + 1))
axes[1].indicate_inset_zoom(ax_insert)

axes[1].set(xlabel="Degree", ylabel="Chi squared", title="Chi squared vs degree")

fig.savefig("figs/full_polynomial_fit.png")


# # Figure
# ax = axes[1]

# ax.plot(range(1, max_degress), chi_squares, marker="o", linestyle="None")

# ax_insert = ax.inset_axes([0.5, 0.5, 0.5, 0.3])

# ax_insert.plot(range(3, max_degress), chi_squares[2:], marker="o", linestyle="None")
# ax_insert.set(ylim=(25, 28))
# ax.indicate_inset_zoom(ax_insert)

# ax.set(xlabel="Degree", ylabel="Chi squared", title="Chi squared vs degree")
