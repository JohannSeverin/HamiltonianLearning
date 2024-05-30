# Standard Solver Options for Quantum Mechanical Simulations
import jax.numpy as jnp
from jax import jit
from diffrax import (
    Dopri5,
    Dopri8,
    SaveAt,
    diffeqsolve,
    DirectAdjoint,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
)


def create_solver(
    t1,
    t0=0,
    adjoint=False,
    stepsize_controller=PIDController(1e-5, 1e-5),
    solver=Dopri5(),
    tlist=None,
    initial_stepsize=1e-1,
    number_of_jump_operators=0,
    max_steps=10000,
):
    adjoint = (
        DirectAdjoint() if adjoint else RecursiveCheckpointAdjoint(checkpoints=None)
    )
    saveat = SaveAt(ts=tlist) if tlist is not None else None

    @jit
    def dynamics(t, rho, args):
        """
        Differential equation governing the system
        """
        drho = get_unitary_term(rho, args["hamiltonian"])

        for i in range(number_of_jump_operators):
            drho += get_dissipation_term(rho, args["jump_operators"][i])

        return drho

    term = ODETerm(dynamics)

    def evolve_states(initial_state, hamiltonian, jump_operators=None):
        return diffeqsolve(
            terms=term,
            solver=solver,
            y0=initial_state,
            t0=t0,
            t1=t1,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            dt0=initial_stepsize,
            args=dict(hamiltonian=hamiltonian, jump_operators=jump_operators),
            adjoint=adjoint,
            max_steps=max_steps,
        )

    return evolve_states


# Functions to compute time evolutions of quantum systems
def get_unitary_term(rho, hamiltonian):
    """
    Compute the unitary term of the Master equation.
    Has to be jax numpy arrays ending with a square matrix as the last term
    """
    return -1j * (
        jnp.einsum("...ij, ...jk -> ...ik", hamiltonian, rho)
        - jnp.einsum("...ij, ...jk -> ...ik", rho, hamiltonian)
    )


def get_dissipation_term(rho, dissipator):
    """
    Compute the dissipator term of the Lindblad equation
    """
    drho = jnp.einsum(
        "...ij, ...jk, ...kl -> ...il", dissipator, rho, dissipator.conj().T
    )

    drho -= 0.5 * (
        jnp.einsum("...ij, ...jk -> ...ik", dissipator.conj().T @ dissipator, rho)
        + jnp.einsum("...ij, ...jk -> ...ik", rho, dissipator.conj().T @ dissipator)
    )

    return drho
