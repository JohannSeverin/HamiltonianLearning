import jax
import jax.numpy as jnp


from diffrax import Dopri5, ODETerm, PIDController, diffeqsolve, SaveAt

from utils import get_unitary_term, get_dissipation_term


# Parameters
QUBIT_ENERGIES = [5.0, 5.2, 5.4]
T1 = 1000.0
TEMPERATURE = 0.050
DURATION = 100
STORE_EVERY = 4
INITIAL_STEPSIZE = 0.1
MAX_SOLVER_STEPS = 100000

# Setup operators
from utils import sigma_z

single_qubit_hamiltonians = [energy * sigma_z() for energy in QUBIT_ENERGIES]


from utils import tensor, identity

id = identity(2)

H0 = (
    tensor(single_qubit_hamiltonians[0], id, id)
    + tensor(id, single_qubit_hamiltonians[1], id)
    + tensor(id, id, single_qubit_hamiltonians[2])
)

# Setup dissipators
rate = 1 / T1

from utils import destroy

dissipators = [jnp.sqrt(rate) * destroy() for _ in QUBIT_ENERGIES]

Ls = [
    tensor(dissipators[0], id, id),
    tensor(id, dissipators[1], id),
    tensor(id, id, dissipators[2]),
]


# Setup of initial states
from itertools import product

from utils import pauli_state_x_dm, pauli_state_y_dm, basis_dm

pauli_z = basis_dm(0, 2)

initial_single_qubit_states = jnp.array(
    list(
        tensor(*state)
        for state in product(
            [pauli_state_x_dm(), pauli_state_y_dm(), pauli_z], repeat=3
        )
    )
)


# Setup of the ODE

from utils import get_unitary_term, get_dissipation_term


def Lindbladian(t, rho, args):
    drho = get_unitary_term(rho, H0)
    for L in Ls:
        drho += get_dissipation_term(rho, L)
    return drho


# Setup of the ODE solver
tlist = jnp.arange(0, DURATION + STORE_EVERY, STORE_EVERY)

term = ODETerm(Lindbladian)
solver = Dopri5()
pid_controller = PIDController(1e-5, 1e-5)

saveat = SaveAt(ts=tlist)

# Solve the ODE
results = diffeqsolve(
    terms=term,
    solver=solver,
    y0=initial_single_qubit_states,
    t0=0,
    t1=DURATION,
    stepsize_controller=pid_controller,
    saveat=saveat,
    dt0=INITIAL_STEPSIZE,
    max_steps=MAX_SOLVER_STEPS,
)

# Plot the results
from utils import sigma_x, sigma_y, sigma_z, identity

id = identity(2)

pauli_operators = dict(x=sigma_x(), y=sigma_y(), z=sigma_z(), I=id)


track = ["x", "I", "I"]
expectation_operator = tensor(*[pauli_operators[op] for op in track])

from utils import expectation_values, rotating_expectation_values

exp_vals = expectation_values(results.ys, expectation_operator)
rot_exp_vals = rotating_expectation_values(
    jnp.transpose(results.ys, axes=[1, 0, 2, 3]), expectation_operator, tlist, H0 * 1.01
).T

import matplotlib.pyplot as plt

for init in range(exp_vals.shape[1]):
    plt.plot(tlist, rot_exp_vals[:, init], label=f"Initial state {init}")
