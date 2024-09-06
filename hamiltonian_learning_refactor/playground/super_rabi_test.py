import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


hamiltonian_params = jnp.array([0.5, 0, 0])
dissipation_params = jnp.zeros((3, 3)).at[2, 2].set(0.01)


chi_generator = jnp.zeros((4, 4), dtype=jnp.complex128)

# Unitary Dynamics
chi_generator = chi_generator.at[0, 1:].add(-1j * hamiltonian_params)
chi_generator = chi_generator.at[1:, 0].add(1j * hamiltonian_params)

# Lower Block
chi_generator = chi_generator.at[1:, 1:].add(dissipation_params)

# Normalization constraint
chi_generator = chi_generator.at[0, 0].add(-jnp.trace(chi_generator))

# TODO: Add complex part of the dissipation matrix to the chi generator


from super_operators import chi_matrix_to_pauli_transfer_matrix
from super_states import density_matrix_to_superket, superket_to_density_matrix

PTM_generator = chi_matrix_to_pauli_transfer_matrix(chi_generator, 1)

# timestep map
dt = 0.01
steps = 2500
PTM_generator = PTM_generator * dt
PTM = jax.scipy.linalg.expm(PTM_generator)


# Define matrix exponentiation jax.lax.scan with carry over


def func(carry, _):
    next_step = PTM @ carry
    return next_step, next_step


final, maps = jax.lax.scan(func, jnp.eye(4, dtype=jnp.complex128), jnp.arange(steps))

times = jnp.arange(steps) * dt

# # Dynamics
# from jax.scipy.linalg import expm

# times = jnp.linspace(0, 10, 1000)

# maps = expm(PTM_generator[None, ...] * times[..., None, None], max_squarings=64)

# Initial state
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from utils.operators import sigma_x, sigma_y, sigma_z, identity

initial_state = (identity(2) + sigma_z(2)) / 2
initial_superket = density_matrix_to_superket(initial_state, 1)

final_superket = jnp.einsum("ijk, ...kl->...ijl", maps, initial_superket).squeeze(0)


# convert back to density matrix
final_density_matrix = superket_to_density_matrix(final_superket, 1)
for op_name, op in {
    "identity": identity(2),
    "sigma_x": sigma_x(2),
    "sigma_y": sigma_y(2),
    "sigma_z": sigma_z(2),
}.items():
    exp_vals = jnp.einsum("ijk, kj->i", final_density_matrix, op)
    plt.plot(times, exp_vals, label=op_name)
